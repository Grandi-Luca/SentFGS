from transformers.generation_stopping_criteria import StoppingCriteria
import torch

class MultiBatchEndSentenceCriteria(StoppingCriteria):
    """
    stop generation after generation if all last tokens is pad_token_ids

    Args:
    
        num_sents: stop generation after specified number of sentences is generated   # deprecate
        
    """
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor) -> bool:
        return (input_ids[:,-1]==self.pad_token_id).all().item()
