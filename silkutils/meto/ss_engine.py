import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import meto.ss_meto as ss_meto

class Engine:
    def __init__(self, discrete_bins, mode, debugging, meta_init_data, structure_limit):
        self.discrete_bins = discrete_bins
        self.mode = mode
        self.debugging = debugging
        
        self.num_base_tokens = discrete_bins
        
        
        if self.mode==3: # vertex: 3 tokens for xyz
            self.impl = ss_meto.Engine_SilkSong(discrete_bins=discrete_bins, verbose=False, debugging=debugging, meta_init_data=meta_init_data, structure_limit=structure_limit)
            self.num_special_tokens = self.impl.OP.OP_NUM
            self.num_word_table = self.num_base_tokens + self.num_special_tokens + self.impl.topology_num # word table
            self.impl.num_word_table=self.num_word_table
        elif self.mode==4: # vertex: 2 tokens for BO
            self.impl = ss_meto.Engine_SilkSong(discrete_bins=discrete_bins, verbose=False, debugging=debugging, meta_init_data=meta_init_data, structure_limit=structure_limit)
            self.num_special_tokens = self.impl.OP.OP_NUM
            self.blocks=8
            self.offsets=16
            self.num_word_table = self.blocks**3 + self.offsets**3 + self.num_special_tokens + self.impl.topology_num # word table
            self.impl.num_word_table=self.num_word_table
        

    def get_metaData(self):
        return self.impl.get_metaData()
    

    def encode(self, vertices, faces, non_mani_process):
        # vertices: [N, 3], float
        # faces: [M, 3], int
        tokens = self.impl.encode(vertices, faces, non_mani_process)
        return np.asarray(tokens)
    
    def token_encode(self, input_tokens, mode):
        tokens = self.impl.token_encode(input_tokens, mode)
        return np.asarray(tokens), self.impl.get_metaData()

    def decode(self, tokens, discrete_bins, mode, colorful=False, manifix=False):
        # tokens: [N], int
        vertices, faces = self.impl.decode(tokens, discrete_bins, mode, colorful, manifix)
        ret_1=None
        ret_2=None
        if colorful:
            ret_1=[np.asarray(ele) for ele in vertices]
        else:
            ret_1=np.asarray(vertices)
        if manifix:
            ret_2=[np.asarray(ele) for ele in faces]
        else:
            ret_2=np.asarray(faces)
        return ret_1, ret_2
    
    def decode_ori(self, tokens):
        # tokens: [N], int
        vertices, faces= self.impl.decode_ori(tokens)
        return np.asarray(vertices), np.asarray(faces)
    
    def decode_bfs(self, faces):
        # tokens: [N], int
        vertices, faces= self.impl.decode_bfs(faces)
        return np.asarray(vertices), np.asarray(faces)
    
    def translate_tokens(self, tokens, discrete_bins, mode):
        translated_tokens=self.impl.translate_tokens(tokens, discrete_bins, mode)
        return translated_tokens
    
    def translate_tokens_direct(self, tokens):
        translated_tokens=self.impl.translate_tokens_direct(tokens)
        return translated_tokens
    
    def trans_compare_tokens(self, tokens, tokens_gt):
        difference_tokens=self.impl.translate_compare_tokens(tokens, tokens_gt)
        return difference_tokens
    
    def get_token_classify(self):
        return self.impl.get_token_classify()
    
    def get_token_map_GPT(self, token_classify_dic, mode):
        return self.impl.get_token_map_list_GPT(token_classify_dic, mode)