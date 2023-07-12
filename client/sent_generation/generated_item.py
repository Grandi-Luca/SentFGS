from typing import Optional
import torch 

class GenerationItem:
    def __init__(
        self, 
        token_ids: torch.LongTensor, 
        logsum: float, 
        classification_score: Optional[float] = 0, 
        text: Optional[str] = "", 
        num_tokens_generated:Optional[int] = -1, 
        beamsearch_stopped: Optional[bool] = False,
        seq_score: Optional[float] = 0.0,
    ):
        self.token_ids = token_ids
        self.logsum = logsum
        self.classification_score = classification_score
        self.text = text
        self.num_tokens_generated = num_tokens_generated
        self.beamsearch_stopped = beamsearch_stopped
        self.seq_score = seq_score

    def get_avg_log(self):
        return self.logsum / self.num_tokens_generated