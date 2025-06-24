import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from pytorch_custom_utils import save_load
from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, Any
from einops import rearrange, repeat, pack
from x_transformers import Decoder
from x_transformers.x_transformers import LayerIntermediates
from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
)
from .miche_conditioner import PointConditioner
from functools import partial
from tqdm import tqdm
from silkutils.silksong_tokenization import get_tokenizer_silksong

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def divisible_by(num, den):
    return (num % den) == 0

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)


# main class of auto-regressive Transformer 
@save_load()
class SSMeshTransformer(Module):
    @beartype
    def __init__(
        self,
        *,
        dim: Union[int, Tuple[int, int]] = 512,  # hidden size of Transformer
        max_seq_len = 9600,                      # max sequence length
        flash_attn = True,                       # wether to use flash attention
        attn_depth = 12,                         # number of layers
        attn_dim_head = 64,                      # dim for each head
        attn_heads = 16,                         # number of heads
        attn_kwargs: dict = dict(
            ff_glu = True,
            num_mem_kv = 4,
            attn_qk_norm = True,
        ),
        dropout = 0.,
        pad_id = -1,
        coor_continuous_range = (-1., 1.),
        num_discrete_coors = 128,
        block_size = 8,
        offset_size = 16,
        mode = 'vertices',
        special_token = -2,
        use_special_block = False,
        conditioned_on_pc = False,
        encoder_name = 'miche-256-feature',
        encoder_freeze = True,
    ):
        super().__init__()

        # if use_special_block:
        #     # block_ids, offset_ids, special_block_ids
        #     vocab_size = block_size**3 + offset_size**3 + block_size**3
        #     self.sp_block_embed = nn.Parameter(torch.randn(1, dim))
        # else:
        #     # block_ids, offset_ids, special_token
        #     vocab_size = block_size**3 + offset_size**3 + 1
        #     self.special_token = special_token
        #     self.special_token_cb = block_size**3 + offset_size**3

        self.tokenizer, vocab_size = get_tokenizer_silksong(resolution=num_discrete_coors, ss_mode=4)
        self.matrix_in0_embed=nn.Parameter(torch.randn(1, dim))
        self.matrix_in1_embed=nn.Parameter(torch.randn(1, dim))
        self.matrix_out_embed=nn.Parameter(torch.randn(1, dim))

        self.block_embed = nn.Parameter(torch.randn(1, dim))
        self.offset_embed = nn.Parameter(torch.randn(1, dim))


        # self.use_special_block = use_special_block
        
        self.sos_token = nn.Parameter(torch.randn(dim))
        self.eos_token_id = vocab_size
        self.pad_in_table = vocab_size + 1
        self.mode = mode
        self.token_embed = nn.Embedding(vocab_size + 2, dim) # v+1:padding; 1-vocab_size (mesh); vocab_size=eos
        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range
        self.block_size = block_size
        self.offset_size = offset_size
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len
        self.conditioner = None
        self.conditioned_on_pc = conditioned_on_pc
        cross_attn_dim_context = None
        
        assert self.block_size * self.offset_size == self.num_discrete_coors

        # load point_cloud encoder
        if conditioned_on_pc:
            print(f'Point cloud encoder: {encoder_name} | freeze: {encoder_freeze}')
            self.conditioner = PointConditioner(model_name=encoder_name, freeze=encoder_freeze)
            cross_attn_dim_context = self.conditioner.dim_latent
        else:
            raise NotImplementedError
        
        # main autoregressive attention network
        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            cross_attend = conditioned_on_pc,
            cross_attn_dim_context = cross_attn_dim_context,
            cross_attn_num_mem_kv = 4,  # needed for preventing nan when dropping out text condition
            **attn_kwargs
        )

        self.to_logits = nn.Linear(dim, vocab_size + 2)
        self.pad_id = pad_id # -1
        
        self.ind_control=0
        self.ind_I0= 3
        self.ind_I1= 3 + self.tokenizer.impl.layer_vertex_limit
        self.ind_Out= 3 + self.tokenizer.impl.inlayer_emb_num
        self.ind_Block= 3 + self.tokenizer.impl.topology_num
        self.ind_Offset= self.ind_Block + self.block_size**3
        self.ind_eos = vocab_size
    
    def get_block_mask(self, codes):
        return (self.ind_Block <= codes) & (codes < self.ind_Offset)
    
    def get_offset_mask(self, codes):
        return (self.ind_Offset <= codes) & (codes < self.ind_eos)
    
    def get_I0_mask(self, codes):
        return (self.ind_I0 <= codes) & (codes < self.ind_I1)
    
    def get_I1_mask(self, codes):
        return (self.ind_I1 <= codes) & (codes < self.ind_Out)
    
    def get_Out_mask(self, codes):
        return (self.ind_Out <= codes) & (codes < self.ind_Block)
    


    @property
    def device(self):
        return next(self.parameters()).device


    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        prompt: Optional[Tensor] = None,
        pc: Optional[Tensor] = None,
        cond_embeds: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        return_codes = True,
        cache_kv = True,
        max_seq_len = None,
        face_coords_to_file: Optional[Callable[[Tensor], Any]] = None,
        tqdm_position = 0,
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if exists(prompt):
            assert not exists(batch_size)

            prompt = rearrange(prompt, 'b ... -> b (...)')
            assert prompt.shape[-1] <= self.max_seq_len

            batch_size = prompt.shape[0]

        # encode point cloud
        if cond_embeds is None:
            if self.conditioned_on_pc:
                cond_embeds = self.conditioner(pc = pc)
        
        assert cond_embeds is not None

        batch_size = default(batch_size, 1)

        codes = default(prompt, torch.empty((batch_size, 0), dtype = torch.long, device = self.device))

        curr_length = codes.shape[-1]

        cache = None

        # predict tokens auto-regressively
        for i in tqdm(range(curr_length, max_seq_len), position=tqdm_position, 
                      desc=f'Process: {tqdm_position}', dynamic_ncols=True, leave=False):

            output = self.forward_on_codes(
                codes,
                return_loss = False,
                return_cache = cache_kv,
                append_eos = False,
                cond_embeds = cond_embeds,
                cache = cache
            )

            if cache_kv:
                logits, cache = output

            else:
                logits = output

            # sample code from logits
            logits = logits[:, -1]
            filtered_logits = filter_logits_fn(logits, **filter_kwargs)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)
            codes, _ = pack([codes, sample], 'b *')

            # check for all rows to have [eos] to terminate

            is_eos_codes = (codes == self.eos_token_id)

            if is_eos_codes.any(dim = -1).all():
                break


        # mask out to padding anything after the first eos

        mask = is_eos_codes.float().cumsum(dim = -1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)  # self.pad_id==-1
        
        # early return of raw residual quantizer codes

        if return_codes:
            # codes = rearrange(codes, 'b (n q) -> b n q', q = 2)
            # if not self.use_special_block:
            #     codes[codes == self.special_token_cb] = self.special_token # self.special_token_cb=B^3+O^3; special_token=-2
            return codes
        else:
            raise Exception('please return codes')
       


    def forward(
        self,
        *,
        codes:          Optional[Tensor] = None,
        cache:          Optional[LayerIntermediates] = None,
        **kwargs
    ):
        # convert special tokens
        # if not self.use_special_block:
        #     codes[codes == self.special_token] = self.special_token_cb 
            
        return self.forward_on_codes(codes, cache = cache, **kwargs)


    def forward_on_codes(
        self,
        codes = None,
        return_loss = True,
        return_cache = False,
        append_eos = True,
        cache = None,
        pc = None,
        cond_embeds = None,
    ):
        # handle conditions

        attn_context_kwargs = dict()
        
        if self.conditioned_on_pc:
            assert exists(pc) ^ exists(cond_embeds), 'point cloud should be given'
            
            # preprocess faces and vertices
            if not exists(cond_embeds):
                cond_embeds = self.conditioner(
                    pc = pc,
                    pc_embeds = cond_embeds,
                )
            
            attn_context_kwargs = dict(
                context = cond_embeds,
                context_mask = None,
            )

        # take care of codes that may be flattened

        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        # prepare mask for position embedding of block and offset tokens
        I0_mask=self.get_I0_mask(codes)
        I1_mask=self.get_I1_mask(codes)
        block_mask=self.get_block_mask(codes)
        offset_mask=self.get_offset_mask(codes)
        Out_mask=self.get_Out_mask(codes)
        

        # get some variable

        batch, seq_len, device = *codes.shape, codes.device

        assert seq_len <= self.max_seq_len, \
            f'received codes of length {seq_len} but needs to be less than {self.max_seq_len}'

        # auto append eos token, based on token padding

        if append_eos:
            assert exists(codes)

            code_lens = ((codes == self.pad_id).cumsum(dim = -1) == 0).sum(dim = -1)

            codes = F.pad(codes, (0, 1), value = -1)  # value=-1

            batch_arange = torch.arange(batch, device = device)

            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')

            codes[batch_arange, code_lens] = self.eos_token_id


        # if returning loss, save the labels for cross entropy

        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes

        # token embed

        codes = codes.masked_fill(codes == self.pad_id, self.pad_in_table) # -1 ---> pad_in_table
        codes = self.token_embed(codes)

        # codebook embed + absolute positions

        seq_arange = torch.arange(codes.shape[-2], device = device)
        codes = codes + self.abs_pos_emb(seq_arange)
        
        # add positional embedding for block and offset token
        block_embed = repeat(self.block_embed, '1 d -> b n d', n = seq_len, b = batch)
        offset_embed = repeat(self.offset_embed, '1 d -> b n d', n = seq_len, b = batch)
        I0_embed = repeat(self.matrix_in0_embed, '1 d -> b n d', n = seq_len, b = batch)
        I1_embed = repeat(self.matrix_in1_embed, '1 d -> b n d', n = seq_len, b = batch)
        Out_embed = repeat(self.matrix_out_embed, '1 d -> b n d', n = seq_len, b = batch)
        codes[block_mask] += block_embed[block_mask]
        codes[offset_mask] += offset_embed[offset_mask]
        codes[I0_mask] += I0_embed[I0_mask]
        codes[I1_mask] += I1_embed[I1_mask]
        codes[Out_mask] += Out_embed[Out_mask]
        
        # if self.use_special_block:
        #     sp_block_embed = repeat(self.sp_block_embed, '1 d -> b n d', n = seq_len, b = batch)
        #     codes[sp_block_mask] += sp_block_embed[sp_block_mask]

        # auto prepend sos token

        sos = repeat(self.sos_token, 'd -> b d', b = batch)
        codes, _ = pack([sos, codes], 'b * d')

        # attention

        attended, intermediates_with_cache = self.decoder(
            codes,
            cache = cache,
            return_hiddens = True,
            **attn_context_kwargs
        )

        # logits

        logits = self.to_logits(attended)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, intermediates_with_cache

        # loss

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.pad_id
        )
        
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            raise ValueError("Loss is nan or inf")

        return ce_loss
