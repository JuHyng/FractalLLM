import argparse

from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field

from typing import List, Optional

@dataclass
class ModelArguments:
    """
    Model Arguments.
    """
    model_name: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        metadata={"help": "Pre-trained model name or path"},
    )
    cache_dir: str = field(
        default="./cache_dir",
        metadata={"help": "Cache directory for the pre-trained model"},
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map for model loading ('auto', None, or custom map)"},
    )
    decode_method: str = field(
        default="baseline",
        metadata={"help": "Enable DraftLlamaDecoderLayer replacement"},
    )
    decomp_method: str = field(
        default="weight_svd",
        metadata={"help": "Decomposition method for DraftLlamaDecoderLayer"},
    )
        
    num_draft_layers: int = field(
        default=7,
        metadata={"help": "Number of DraftLlamaDecoderLayer to replace"},
    )
    draft_layer_indexes: List[int] = field(
        default_factory=list,
        metadata={"help": "Layer indexes to replace with DraftLlamaDecoderLayer"},
    )
    
    layers_dict_json: str= field(
        default=None,
        metadata={"help": "json file of layers and weights to be decomposed"}
    )
    
    layers_gap: int = field(
        default=1,
        metadata={"help": "gap between layers to be decomposed"},
    )
    
    rank : Optional[int] = field(
        default=1,
        metadata={"help": "rank of SVD decomposition"},
    )
    
    saved_dir: str = field(
        default="./cache_dir/decomposed_weights/Llama-2-7b-hf",
        metadata={"help": "directory to save the decomposed weights"},
    )
    
    to_save: bool = field(
        default=True,
        metadata={"help": "save the decomposed weights"},
    )

    draft_token: str = field(
        default="independent",
        metadata={"help": "Draft token for input)"},
    )
    draft_len: int = field(
        default=4,
        metadata={"help": "Bucket size for 'bucket' or 'independent' masking (default: 4)"},
    )
    
    print_draft: bool = field(
        default=False,
        metadata={"help": "Print results during drafting"},
    )
    
    sweep: bool = field(
        default=False,
        metadata={"help": "Sweep the model"},
    )
    
@dataclass
class DataArguments:
    """
    Data Arguments.
    """
    dataset: str = field(
        default="gsm8k",
        metadata={"help": "Dataset name (default: gsm8k)"},
    )
    split: str = field(
        default="train",
        metadata={"help": "Dataset split to use ('train', 'test', etc.)"},
    )
    max_samples: int = field(
        default=5,
        metadata={"help": "Maximum number of samples to process (default: 5)"},
    )
    noise_std: float = field(
        default=0.1,
        metadata={"help": "Standard deviation of Gaussian noise for 'noise' mode"},
    )
    
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for training/evaluation"},
    )
    
    n_fewshot: int = field(              
        default=0,
        metadata={"help": "0이면 zero-shot, k>0 few-shot으로 prepend"},
    )
    

@dataclass
class GenerationArguments:
    """
    Generation Config Arguments.
    """
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the sequence to be generated."},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "Sampling temperature."},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "Top-K sampling."},
    )
    top_p: Optional[float] = field(
        default=1.0,
        metadata={"help": "Top-P (nucleus) sampling."},
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use sampling."},
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search."},
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Penalty for repetition."},
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Length penalty for beam search."},
    )
    early_stopping: Optional[bool] = field(
        default=False,
        metadata={"help": "Stop early when using beam search."},
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={"help": "N-gram size to avoid repetition."},
    )
    use_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "Use cache for faster decoding."},
    )

from dataclasses import dataclass, field

@dataclass
class LayerSweepArguments:
    """
    Boolean flags (0/1) indicating whether each layer is used in draft routing.
    Up to 8 layers will be sampled in main_sweep.py.
    """
    layer_0:  int = field(default=0, metadata={"help": "Use layer 0 in draft routing"})
    layer_1:  int = field(default=0)
    layer_2:  int = field(default=0)
    layer_3:  int = field(default=0)
    layer_4:  int = field(default=0)
    layer_5:  int = field(default=0)
    layer_6:  int = field(default=0)
    layer_7:  int = field(default=0)
    layer_8:  int = field(default=0)
    layer_9:  int = field(default=0)
    layer_10: int = field(default=0)
    layer_11: int = field(default=0)
    layer_12: int = field(default=0)
    layer_13: int = field(default=0)
    layer_14: int = field(default=0)
    layer_15: int = field(default=0)
    layer_16: int = field(default=0)
    layer_17: int = field(default=0)
    layer_18: int = field(default=0)
    layer_19: int = field(default=0)
    layer_20: int = field(default=0)
    layer_21: int = field(default=0)
    layer_22: int = field(default=0)
    layer_23: int = field(default=0)
    layer_24: int = field(default=0)
    layer_25: int = field(default=0)
    layer_26: int = field(default=0)
    layer_27: int = field(default=0)
    layer_28: int = field(default=0)
    layer_29: int = field(default=0)
    layer_30: int = field(default=0)
    layer_31: int = field(default=0)


def get_args():
    
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, GenerationArguments, LayerSweepArguments))
    
    model_args, training_args, data_args, gen_args, layer_sweep_args= parser.parse_args_into_dataclasses()
    @dataclass
    class CombinedArgs:
        pass
    
    combined_args = CombinedArgs()
    
    for obj in [model_args, training_args, data_args, gen_args, layer_sweep_args]:
        for key, value in vars(obj).items():
            setattr(combined_args, key, value)
    
    return combined_args
