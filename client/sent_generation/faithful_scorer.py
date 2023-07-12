import os
import torch
from typing import Any, Optional, Tuple, List
import spacy
spacy.prefer_gpu()
import json

from sentence_utils import ConverterSentenceToAMR
from metrics import AMRScorer, FactGraph

from factgraph.process_docu_amr import process
import factgraph.augmentation_ops as ops

class Scorer():
    def __init__(
        self, 
        tokenizer, 
        amrs_input_path_name: str,
        converter_to_amr: ConverterSentenceToAMR,
        metric: AMRScorer, 
    ):
        assert not isinstance(metric, FactGraph), 'if you want to use fact graph as metric selelct FactGraphScorer class'
        self._tokenizer = tokenizer
        self._amrs_input_path_name = amrs_input_path_name
        self._metric = metric
        self._converter_to_amr = converter_to_amr
        self._nlp = spacy.load('en_core_web_sm',exclude=["tagger","entity_linker","entity_ruler","textcat",
                                                          "textcat_multilabel","attribute_ruler", "lemmatizer",
                                                          "trainable_lemmatizer","morphologizer","attribute_ruler",
                                                          "transformer","ner","senter","sentencizer"])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, input_text = None) -> torch.FloatTensor:
        # decode the sequence and split into sentences
        for idx, hyp in enumerate(self._tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
            beam_sentences = [sent.text for sent in self._nlp(hyp).sents]
            scores[idx] = 0
            
            if len(beam_sentences) > 0:
                # convert sentence to its amr graph
                sentence_amr_location = self._converter_to_amr([beam_sentences[-1]])

                # check if the path of input's amr graphs is a directory
                if(os.path.isdir(self._amrs_input_path_name)):
                    preds = []
                    # iterate over files in specified directory
                    for filename in os.listdir(self._amrs_input_path_name):
                        f = os.path.join(self._amrs_input_path_name, filename)
                        if os.path.isfile(f) and os.path.realpath(f) != sentence_amr_location and ".amr" in f:
                            preds.extend(self._metric.predict_score(sentence_amr_location, f))
                    final_pred_score = max(preds)
                
                else:
                    final_pred_score = max(self._metric.predict_score(sentence_amr_location, self._amrs_input_path_name))

                # execute similarity matching and update final score
                scores[idx] = final_pred_score

        return scores

class FactGraphScorer():
    def __init__(
        self, 
        tokenizer, 
        amrs_input_path_name: str,
        metric: FactGraph, 
        converter_to_amr: ConverterSentenceToAMR,
        selectSentence = None,
        processed_amrs = None,
    ):
        self._tokenizer = tokenizer
        self._amrs_input_path_name = amrs_input_path_name
        self._metric = metric
        self._converter_to_amr = converter_to_amr
        self.selectSentence = selectSentence
        self.processed_amrs = processed_amrs
        self._nlp = spacy.load('en_core_web_sm',exclude=["tagger","entity_linker","entity_ruler","textcat",
                                                          "textcat_multilabel","attribute_ruler", "lemmatizer",
                                                          "trainable_lemmatizer","morphologizer","attribute_ruler",
                                                          "transformer","ner","senter","sentencizer"])
    
    def create_and_preprocess_input_graph(self, input_text: str, amr_path):
        # create input graphs
        page_doc = self._nlp(input_text)
        sents = [sent for sent in page_doc.sents]
        sents_text = [sent.text for sent in sents]
        for idx, t in enumerate(sents_text):
            amr_loc = self._converter_to_amr([t], f'{idx}-input_text')
            if not os.path.exists(amr_loc):
                raise Exception(amr_loc)
            
        self.preprocess_input_graph(input_text, len(sents_text), amr_path)


    def preprocess_input_graph(self, input_text: str, num_sentences: int, amr_path: str):
        # preprocess input graphs
        self.processed_amrs = process(amr_path, num_sentences, self._metric.tokenizer)

        self.selectSentence = ops.SelectSentencesScore(input_text)

    def __call__(self, input_ids, scores, input_text, multi_sentence_eval: Optional[bool] = None):
        alldata = []
        summ_sents = []
        sent_id = 0
        for idx, hyp in enumerate(self._nlp.pipe(self._tokenizer.batch_decode(input_ids, skip_special_tokens=True))):
            scores[idx] = 0
            beam_sentences = [sent.text for sent in hyp.sents if len(sent) > 0]
            beam_sentences = beam_sentences if bool(multi_sentence_eval) else beam_sentences[-1:]
            for sent in beam_sentences:
                summ_sents.append(sent)
                example = {}
                example["hyp_id"] = idx
                example["sent_id"] = sent_id
                example["article"] = input_text
                example["summary"] = sent
                example = self.selectSentence.transform(example, self._metric.num_document_graphs)
                alldata.append(example)
                sent_id+=1

        output_file_name = "summ_amrs"
        sentence_amr_location = self._converter_to_amr(summ_sents, output_file_name)

        if not os.path.exists(sentence_amr_location):
            raise Exception(sentence_amr_location)
        amr_graphs = {}
        for d in alldata:
            selected_amr_graphs = [self.processed_amrs[s[0]] for s in json.loads(d["sentences"]) if s[0] in self.processed_amrs]
            amr_graphs[d["sent_id"]] = selected_amr_graphs
        if len(alldata) != 0:
            output = self._metric.predict_score(alldata,sentence_amr_location,amr_graphs)
            for k in output:
                scores[k] = output[k]

        return scores