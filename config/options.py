import tyro
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal

@dataclass
class MetoConfigs:
    """Mesh to token engine config"""
    # basic
    discrete_bins: int = 128
    ss_mode: int = 4
    block_size: int = 8
    offset_size: int = 16


@dataclass
class TrainConfigs:
    # basic
    lr: float = 1e-4
    num_epochs: int = 100
    save_epoch: int = 1
    eval_mode: str = "loss" # loss / none
    warmup_ratio: float = 0.01 
    use_wandb: int = 1

    # resume
    resume: Optional[str] = None
    resume_epoch: int = 0
    ft: int = 0

    # others
    resume_step_ratio: float = 0
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    mixed_precision: Literal['no', 'fp8', 'fp16', 'fp32'] = 'fp16'
    checkpointing: bool = True # gradient checkpointing
    debug_eval: int = 0

@dataclass
class InferConfigs:
    # basic
    test_path_input: str = "test_datasets/sb06"
    test_repeat: int = 1
    infer_batch: int = 1
    temperature: float = 0.5
    max_filter: int = 0
    

@dataclass
class DataConfigs:
    
    # dataset
    dataset: str = "ss" # debug_one / ss
    data_subsets: str = "gobjaversev1"
    xlsx_dir: str = "datasets/cleaned"
    testset_xlsx_dir: str = "datasets/sample_test/tables"
    testset_prefix: str = "testset"
    data_filter_cnt: int = 8

    # resample
    resample: int = 0
    face_delta: int = 100
    i_beta: float=0.0
    e_beta: float=1.0

    # iter
    batch_size: int = 4
    num_workers: int = 4
    testset_size: int = 32

    # aug
    use_scale_aug: int = 1
    use_rot_aug: int = 1
    use_decimate_aug: int = 0
    

@dataclass
class ModelConfigs:

    # encoder
    conditioned_on_pc: int = 1
    encoder_name: str = "miche-256-feature"
    encoder_freeze: int = 0
    pc_num: int = 4096

    # GPT
    mode: str = "vertices"
    dim: int = 1024
    depth: int = 24
    attn_dim_head: int = 64
    attn_heads: int = 16
    dropout: float = 0.0
    pad_token_id: int = -1



@dataclass
class LoggingConfigs:

    output_dir: str = "outputs"
    log_dir: str = "logs"
    

@dataclass
class AllConfigs:
    
    train: TrainConfigs = field(default_factory=TrainConfigs)
    infer: InferConfigs = field(default_factory=InferConfigs)
    data: DataConfigs = field(default_factory=DataConfigs)
    model: ModelConfigs = field(default_factory=ModelConfigs)
    logging: LoggingConfigs = field(default_factory=LoggingConfigs)
    meto: MetoConfigs = field(default_factory=MetoConfigs)
    
    # global
    workspace: str = "workspace_train/"
    max_face_length: int = 12000
    max_seq_length: int = 10240
    seed: int = 0

    def __getattr__(self, name):
        
        for config in [self.train, self.infer, self.data, self.model, self.logging, self.meto]:
            if hasattr(config, name):
                return getattr(config, name)
        raise AttributeError(f"'AllConfigs' object has no attribute '{name}'")
    
    def __post_init__(self):
        # post process
        if self.data.batch_size <= 0:
            raise ValueError("batch_size must be positive")
