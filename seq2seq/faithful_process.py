import os
import torch
from typing import Any, Optional
from transformers.generation_logits_process import LogitsProcessor
from seq2seq.metrics import Metric, WWLK
from seq2seq.sentence_utils import ConverterSentenceToAMR, SpringConverter

import spacy


def to_sentences(doc:str, nlp = None):
    sents = doc.strip().split('\n')
    if(nlp is not None):
        return [sent.text for s in sents for sent in nlp(s).sents if len(sent.text) > 0]
    else:
        return []

class FaithfulProcess(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that increase the score of faithful output sequences
    Args:
        tokenizer:
            Tokernizer too decode output sequences.
        amrs_input_path_name (`str`):
            Path of the file that contains all the amr graphs of the input sequence
        metric (`Metric`, *optional*, defaults wasser weisfeiler leman [WWLK]):
            Metric use to calculate similarity metching
        faithful_penalty (`float`, *optional*, defaults 0.0):
            Value that indicates how strong is the penalty applied
    """

    def __init__(
        self, 
        tokenizer, 
        amrs_input_path_name: str,
        metric: Optional[Metric] = None, 
        faithful_penalty: Optional[float] = None,
        converter_to_amr: Optional[ConverterSentenceToAMR] = None,
    ):
        self._tokenizer = tokenizer
        self._amrs_input_path_name = amrs_input_path_name
        self._metric = metric if metric is not None else WWLK()
        self._faithful_penalty = faithful_penalty if faithful_penalty is not None else 0.0
        self._converter_to_amr = converter_to_amr if converter_to_amr is not None else SpringConverter()

        self._nlp = spacy.load('en_core_web_sm')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, seq_starts: torch.LongTensor, seq_ends: torch.LongTensor) -> torch.FloatTensor:
        # decode the sequence and split into sentences
        for idx, hyp in enumerate(input_ids):
            if seq_ends[idx] != -1:
                beam_sentences = to_sentences(self._tokenizer.decode(hyp, skip_special_tokens=True)[seq_starts[idx]:seq_ends[idx]], self._nlp)
            else:
                beam_sentences = to_sentences(self._tokenizer.decode(hyp, skip_special_tokens=True)[seq_starts[idx]:], self._nlp)

            # convert sentence to amr
            for sentence in beam_sentences:
                sentence_amr_location = self._converter_to_amr(sentence)

                # check if the path of input's amr graphs is a directory
                if(os.path.isdir(self._amrs_input_path_name)):
                    preds = []
                    # iterate over files in specified directory
                    for filename in os.listdir(self._amrs_input_path_name):
                        f = os.path.join(self._amrs_input_path_name, filename)
                        if os.path.isfile(f) and f != sentence_amr_location and ".amr" in f:
                            preds.extend(self._metric.predict_score(sentence_amr_location, f))
                    final_pred_score = max(preds)
                
                else:
                    final_pred_score = max(self._metric.predict_score(sentence_amr_location, self._amrs_input_path_name))

                # execute similarity matching and update final score
                scores[idx] = scores[idx] + self._faithful_penalty * final_pred_score

        return scores