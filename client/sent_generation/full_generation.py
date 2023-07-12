from typing import Tuple, List, Optional, Union

import torch

from transformers import (
    set_seed,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BeamSearchScorer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
)

from stopping_criteria import ( # self-defined
    EndSpanCriteria,
    MultiBatchEndSentenceCriteria,
)

from generated_item import GenerationItem
from sentences_generation import SentGenerator

class FullSequenceGenerator():
    def __init__(
        self, 
        model,
        tokenizer, 
        scorer = None,
        fact_penalty: Optional[float] = None, 
        seed: Optional[int] = None,
    ) -> None:
        self.generator = SentGenerator(model, tokenizer)
        self.tokenizer = tokenizer
        if seed is not None:
            set_seed(seed)
        self.scorer = scorer
        self.fact_penalty = fact_penalty if fact_penalty is not None else 0

    # # --------- Generation Functions ---------
    def process_beamsearch_generation(self, beamsearch_outputs, start_pos, prev_gen: Optional[GenerationItem] = None):
        """
        return:
            tuple (GenerationItem, beamsearch_stopped)
            if no additional sentence is generated, returned GenerationItem is None, beamsearch_stopped is True 
        """
        # NOTE: 2 dim, with first dim of size 1 in order to work
        assert beamsearch_outputs.sequences.size(0) == 1 and beamsearch_outputs.sequences.dim() == 2
        # cut off pad ids
        pad_mask = beamsearch_outputs.sequences==self.tokenizer.pad_token_id
        eos_mask = beamsearch_outputs.sequences==self.tokenizer.eos_token_id
        comb_mask = pad_mask.logical_or(eos_mask)
        last_valid_idx = (comb_mask == False).nonzero()[-1][1].item() 
        curr_gen_ids = beamsearch_outputs.sequences[:,:last_valid_idx+1]
        
        # only add to generations if new sentence generated
        # future beam search will be skipped if no new sentence generated from previous beam search
        if prev_gen is None or curr_gen_ids.size(1) > start_pos: 
            # get scores
            end_pos = last_valid_idx+1 # later put pad token probability to be 1 
            gen_ids = beamsearch_outputs.sequences[:, start_pos:end_pos][0]
            num_tokens_generated = end_pos - start_pos
            logsum = beamsearch_outputs.sequences_scores.item() * (num_tokens_generated**2)
            # classification score
            new_sent = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if self.fact_penalty <= 0 or self.scorer is None:
                fact_scores = torch.tensor([0], dtype=torch.float, device=beamsearch_outputs.sequences.device)
            else:
                fact_scores = self.scorer(gen_ids, beamsearch_outputs.sequences_scores, self.generator.context)
            # finalize value with prev_gen
            prev_gen_num_tokens = prev_gen.num_tokens_generated if prev_gen is not None else 0
            prev_gen_logsum = prev_gen.logsum if prev_gen is not None else 0
            prev_gen_text = prev_gen.text if prev_gen is not None else ""
            text = " ".join([prev_gen_text.strip(), new_sent.strip()]) 
            logsum += prev_gen_logsum
            num_tokens_generated += prev_gen_num_tokens
            prev_gen_class_score = prev_gen.classification_score if prev_gen is not None else 0
            fact_scores[0] += prev_gen_class_score
            item = GenerationItem(curr_gen_ids, logsum, fact_scores[0].item(), text, num_tokens_generated, beamsearch_stopped=False)
            return item

    def process_beamsample_generation(self, beamsearch_outputs, start_pos, prev_gen: Optional[GenerationItem] = None):
        """
        return:
            tuple (List[GenerationItem], beamsearch_stopped)
            if no additional sentence is generated, returned GenerationItem is None, beamsearch_stopped is True 
        """

        assert beamsearch_outputs.sequences.dim() == 2
        generations = []
        gen_indices = []
        for gen_idx in range(beamsearch_outputs.sequences.size(0)):
            # cut off pad ids
            pad_mask = beamsearch_outputs.sequences[gen_idx]==self.tokenizer.pad_token_id
            eos_mask = beamsearch_outputs.sequences[gen_idx]==self.tokenizer.eos_token_id
            comb_mask = pad_mask.logical_or(eos_mask)
            last_valid_idx = (comb_mask == False).nonzero()[-1].item() 
            curr_gen_ids = beamsearch_outputs.sequences[gen_idx,:last_valid_idx+1].unsqueeze(0)
            
            # only add to generations if new sentence generated
            # future beam search will be skipped if no new sentence generated from previous beam search
            if prev_gen is None or curr_gen_ids.size(1) > start_pos: 
                # get scores
                end_pos = last_valid_idx+1 # later put pad token probability to be 1 
                gen_ids = beamsearch_outputs.sequences[gen_idx, start_pos:end_pos]
                num_tokens_generated = end_pos - start_pos
                logsum = beamsearch_outputs.sequences_scores[gen_idx].item() * (num_tokens_generated**2)
                # classification score
                new_sent = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                # finalize value with prev_gen
                prev_gen_num_tokens = prev_gen.num_tokens_generated if prev_gen is not None else 0
                prev_gen_logsum = prev_gen.logsum if prev_gen is not None else 0
                prev_gen_text = prev_gen.text if prev_gen is not None else ""
                text = " ".join([prev_gen_text.strip(), new_sent.strip()]) 
                logsum += prev_gen_logsum
                num_tokens_generated += prev_gen_num_tokens
                item = GenerationItem(token_ids=curr_gen_ids,
                                    logsum=logsum, 
                                    text=text, 
                                    num_tokens_generated=num_tokens_generated, beamsearch_stopped=False)
                generations.append(item)
                gen_indices.append(gen_idx)

        if len(gen_indices) > 0:
            if self.fact_penalty <= 0 or self.scorer is None:
                fact_scores = torch.zeros(len(gen_indices), dtype=torch.float, device=beamsearch_outputs.sequences.device)
            else:
                fact_scores = self.scorer(beamsearch_outputs.sequences[gen_indices], beamsearch_outputs.sequences_scores[gen_indices], self.generator.context)
        
            prev_gen_class_score = prev_gen.classification_score if prev_gen is not None else 0
            for idx in gen_indices:
                generations[idx].classification_score = fact_scores[idx].item() + prev_gen_class_score

        return generations

    def process_multisample_generation(self, sample_outputs, start_pos, prev_gen: Optional[GenerationItem] = None):
        """
        return:
            List [GenerationItem]
        """
        generations = []
        gen_indices = []

        # cut off pad ids
        pad_mask = sample_outputs.sequences==self.tokenizer.pad_token_id
        eos_mask = sample_outputs.sequences==self.tokenizer.eos_token_id
        comb_mask = pad_mask.logical_or(eos_mask)
        probs = torch.stack(sample_outputs.scores, dim=0).softmax(-1)

        # format each sequence into a GenerationItem
        for gen_idx in range(sample_outputs.sequences.size(0)): 
            last_valid_idx = (comb_mask[gen_idx] == False).nonzero()[-1].item() 
            curr_gen_ids = sample_outputs.sequences[gen_idx,:last_valid_idx+1].unsqueeze(0)
            # get scores
            end_pos = last_valid_idx+1 # later put pad token probability to be 1 
            gen_ids = sample_outputs.sequences[gen_idx, start_pos:end_pos]
            num_tokens_generated = end_pos - start_pos
            curr_probs = probs[:, gen_idx, :].squeeze(1)[:num_tokens_generated] # size [seq_len, num_beams, vocab_size]
            gen_probs = torch.gather(curr_probs, -1, gen_ids[:, None]).squeeze(-1)
            logsum = torch.sum(torch.log(gen_probs)).item()
            # classification score
            new_sent = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            # finalize value with prev_gen
            prev_gen_num_tokens = prev_gen.num_tokens_generated if prev_gen is not None else 0
            prev_gen_logsum = prev_gen.logsum if prev_gen is not None else 0
            prev_gen_text = prev_gen.text if prev_gen is not None else ""
            text = " ".join([prev_gen_text.strip(), new_sent.strip()]) 
            logsum += prev_gen_logsum
            num_tokens_generated += prev_gen_num_tokens
            item = GenerationItem(token_ids=curr_gen_ids, 
                                  logsum=logsum, 
                                  text=text,
                                  num_tokens_generated=num_tokens_generated)
            generations.append(item)
            gen_indices.append(gen_idx)
        
        if len(gen_indices) > 0:
            if self.fact_penalty <= 0 or self.scorer is None:
                fact_scores = torch.zeros(len(gen_indices), device=sample_outputs.sequences.device)
            else:
                fact_scores = self.scorer(sample_outputs.sequences[gen_indices], sample_outputs.sequences_scores[gen_indices], self.generator.context)
        
            prev_gen_class_score = prev_gen.classification_score if prev_gen is not None else 0
            for idx in gen_indices:
                generations[idx].classification_score = fact_scores[idx].item() + prev_gen_class_score

        return generations
        
    def process_diversebeamsearch_generation(self, outputs, start_pos, prev_gen: Optional[GenerationItem] = None, length_penalty: Optional[float] = None):
        """
        return:
            List [GenerationItem]
        """

        length_penalty = length_penalty if length_penalty is not None else self.generator.model.config.length_penalty

        assert outputs.sequences.dim() == 2
        generations = []
        gen_indices = []
        for gen_idx in range(outputs.sequences.size(0)):
            # cut off pad ids
            pad_mask = outputs.sequences[gen_idx]==self.tokenizer.pad_token_id
            eos_mask = outputs.sequences[gen_idx]==self.tokenizer.eos_token_id
            comb_mask = pad_mask.logical_or(eos_mask)
            last_valid_idx = (comb_mask == False).nonzero()[-1].item() 
            curr_gen_ids = outputs.sequences[gen_idx,:last_valid_idx+1].unsqueeze(0)
            
            # only add to generations if new sentence generated
            # future beam search will be skipped if no new sentence generated from previous beam search
            if prev_gen is None or curr_gen_ids.size(1) > start_pos: 
                # get scores
                end_pos = last_valid_idx+1 # later put pad token probability to be 1 
                gen_ids = outputs.sequences[gen_idx, start_pos:end_pos]
                num_tokens_generated = end_pos - start_pos
                logsum = outputs.sequences_scores[gen_idx].item() * (num_tokens_generated**length_penalty)
                # classification score
                new_sent = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                # finalize value with prev_gen
                prev_gen_num_tokens = prev_gen.num_tokens_generated if prev_gen is not None else 0
                prev_gen_logsum = prev_gen.logsum if prev_gen is not None else 0
                prev_gen_text = prev_gen.text if prev_gen is not None else ""
                text = " ".join([prev_gen_text.strip(), new_sent.strip()]) 
                logsum += prev_gen_logsum
                num_tokens_generated += prev_gen_num_tokens
                item = GenerationItem(token_ids=curr_gen_ids,
                                     logsum=logsum, 
                                     text=text, 
                                     num_tokens_generated=num_tokens_generated,
                                     beamsearch_stopped=False)
                generations.append(item)
                gen_indices.append(gen_idx)

        if len(gen_indices) > 0:
            if self.fact_penalty <= 0 or self.scorer is None:
                fact_scores = torch.zeros(len(gen_indices), dtype=torch.float, device=outputs.sequences.device)
            else:
                fact_scores = self.scorer(outputs.sequences[gen_indices], outputs.sequences_scores[gen_indices], self.generator.context)

            prev_gen_class_score = prev_gen.classification_score if prev_gen is not None else 0
            for idx in gen_indices:
                generations[idx].classification_score = fact_scores[idx].item() + prev_gen_class_score

        return generations 

    def _generate_sent(
        self,
        input_ids, 
        stopping_criteria,
        **kwargs
    ):
        kwargs = {
            **kwargs, 
            **{"output_scores": True, "return_dict_in_generate":True}
        }
        decoder_input_ids = kwargs.pop('decoder_input_ids', None)
        if decoder_input_ids is None:
            return self.generator.generate(
                input_ids=input_ids, 
                stopping_criteria=stopping_criteria,
                **kwargs
            )
        else:
            return self.generator.generate(
                input_ids=input_ids,
                stopping_criteria=stopping_criteria,
                decoder_input_ids=decoder_input_ids,
                **kwargs
            )

    def generate_sentence_options(
        self,
        sample_size: int, 
        input_ids: torch.LongTensor, 
        num_beams: int,
        max_output_length: int,
        top_p: Optional[float] = None,
        prev_gen: Optional[GenerationItem] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
            sample_size: number of sentences to generate from sampling
            input_ids: input_ids from source
            prev_gen: previously generated sentence class
            target_label: the idx for the intended generation
            decoder_input_ids: directly specify the decoder input ids if using ITSP, containing added prompt tokens
        """
        beamsearch_stopped = False # flag this once beam search cannot generate anymore sentences
        generations = []

        multibatch_stopping_criteria = StoppingCriteriaList()
        multibatch_stopping_criteria.append(MultiBatchEndSentenceCriteria(self.tokenizer.pad_token_id))
        decoder_input_ids = decoder_input_ids if decoder_input_ids is not None else prev_gen.token_ids if prev_gen is not None else None
        decoder_input_id_length = decoder_input_ids.size(1) if decoder_input_ids is not None else 0
        start_pos = decoder_input_ids.size(-1) if decoder_input_ids is not None else prev_gen.token_ids.size(-1) if prev_gen is not None else 1

        if decoder_input_id_length >= max_output_length: # no need to generate further if exceed max length
            item = prev_gen
            item.classification_score = 0
            generations.append(item)
            print(f"generation force stopped due to exceeding max length, you may consider use longer MAX_TARGET_LENGTH", 'red')

        else:
            if prev_gen is None or not prev_gen.beamsearch_stopped:
                kwargs = {
                    **kwargs, 
                    **{"do_sample": False, 'num_return_sequences': 1}
                }
                # beam search
                beamsearch_outputs = self._generate_sent(
                    input_ids, 
                    multibatch_stopping_criteria,
                    num_beams= num_beams,
                    decoder_input_ids=decoder_input_ids,
                    **kwargs
                )
                item, beamsearch_stopped = self.process_beamsearch_generation(beamsearch_outputs, start_pos, prev_gen = prev_gen)
                if not beamsearch_stopped:
                    generations.append(item)

            # neucleus sampling
            if sample_size - len(generations) > 0:
                kwargs = {
                    **kwargs, 
                    **{"do_sample": True, 'num_beams': 1, 'num_return_sequences': sample_size - len(generations)}
                }
                sample_outputs = self._generate_sent(
                    input_ids, 
                    multibatch_stopping_criteria,
                    top_p=top_p,
                    decoder_input_ids=decoder_input_ids,
                    **kwargs
                )
                items = self.process_multisample_generation(sample_outputs, start_pos, prev_gen = prev_gen)
                generations.extend(items)

        return (generations, beamsearch_stopped)

    def generate_beamsample_options(
        self,
        sample_size: int,
        input_ids: torch.LongTensor,
        num_beams: int,
        max_output_length: int,
        top_p: Optional[float] = None, 
        prev_gen: Optional[GenerationItem] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
            sample_size: number of sentences to generate from sampling
            num_sents: number of new sentences to generate
            input_ids: input_ids from source
            prev_gen: previously generated sentence class
            target_label: the idx for the intended generation
        """
        generations = []
        multibatch_stopping_criteria = StoppingCriteriaList()
        multibatch_stopping_criteria.append(MultiBatchEndSentenceCriteria(self.tokenizer.pad_token_id))
        decoder_input_ids= decoder_input_ids if decoder_input_ids is not None else prev_gen.token_ids if prev_gen is not None else None
        decoder_input_id_length = decoder_input_ids.size(1) if decoder_input_ids is not None else 0
        start_pos = decoder_input_ids.size(-1) if decoder_input_ids is not None else prev_gen.token_ids.size(-1) if prev_gen is not None else 1


        if decoder_input_id_length >= max_output_length: # no need to generate further if exceed max length
            item = prev_gen
            item.classification_score = 0
            generations.append(item)
            print(f"generation force stopped due to exceeding max length, you may consider use longer MAX_TARGET_LENGTH", 'red')

        else:
            kwargs = {
            **kwargs, 
            **{"do_sample": False, 'num_return_sequences': sample_size}
        }
            # beam sampling 
            beamsample_outputs = self._generate_sent(
                input_ids, 
                multibatch_stopping_criteria,
                top_p=top_p,
                num_beams=num_beams,
                decoder_input_ids=decoder_input_ids,
                **kwargs
            )
            items = self.process_beamsample_generation(beamsample_outputs, start_pos, prev_gen=prev_gen)
            generations.extend(items)
        return generations
    
    def generate_groupbeamsearch_options(
        self,
        sample_size: int,
        input_ids: torch.LongTensor,
        num_beams: int,
        num_beam_groups: int,
        max_output_length: int,
        diversity_penalty: Optional[float] = None,
        length_penalty: Optional[float] = None,
        prev_gen: Optional[GenerationItem] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
        ):
        """
            sample_size: number of sentences to generate
            input_ids: input_ids from source
            prev_gen: previously generated sentence class
        """
        generations = []
        multibatch_stopping_criteria = StoppingCriteriaList()
        multibatch_stopping_criteria.append(MultiBatchEndSentenceCriteria(self.tokenizer.pad_token_id))
        decoder_input_ids = decoder_input_ids if decoder_input_ids is not None else prev_gen.token_ids if prev_gen is not None else None
        decoder_input_id_length = decoder_input_ids.size(1) if decoder_input_ids is not None else 0
        start_pos = decoder_input_ids.size(-1) if decoder_input_ids is not None else prev_gen.token_ids.size(-1) if prev_gen is not None else 1

        kwargs = {
            **kwargs, 
            **{"do_sample": False, 'num_return_sequences':sample_size}
        }

        if decoder_input_id_length >= max_output_length: # no need to generate further if exceed max length
            item = prev_gen
            item.classification_score = 0
            generations.append(item)
            print(f"generation force stopped due to exceeding max length, you may consider use longer MAX_TARGET_LENGTH", 'red')

        else:
            # beam sampling 
            outputs = self._generate_sent(
                input_ids, 
                multibatch_stopping_criteria,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                max_length=max_output_length,
                decoder_input_ids=decoder_input_ids,
                length_penalty=length_penalty,
                **kwargs,
            )
            items = self.process_diversebeamsearch_generation(outputs, start_pos, prev_gen=prev_gen, length_penalty=length_penalty)
            generations.extend(items)
        return generations
        
    def generte_sample_options(
        self,
        sample_size: int, 
        input_ids: torch.LongTensor, 
        max_output_length: int,
        top_p: Optional[float] = None, 
        prev_gen: Optional[GenerationItem] = None,
        **kwargs
    ):
        # neucleus sampling
        decoder_input_id_length = prev_gen.token_ids.size(1) if prev_gen is not None else 0
        start_pos = prev_gen.token_ids.size(-1) if prev_gen is not None else 1

        if decoder_input_id_length >= max_output_length: # no need to generate further if exceed max length
            item = prev_gen
            item.classification_score = 0
            return [item]

        multibatch_stopping_criteria = StoppingCriteriaList()
        multibatch_stopping_criteria.append(MultiBatchEndSentenceCriteria(self.tokenizer.pad_token_id))
        kwargs = {
            **kwargs, 
            **{"do_sample": True, 'num_beams': 1, 'num_return_sequences': sample_size}
        }
        sample_outputs = self._generate_sent(
            input_ids, 
            multibatch_stopping_criteria,
            top_p=top_p,
            decoder_input_ids=prev_gen.token_ids if prev_gen is not None else None,
            **kwargs
        )
        items = self.process_multisample_generation(sample_outputs, start_pos, prev_gen)
        return items

    def sort_filter_gen_history(self, sent_options:List[GenerationItem], n:int): # n is the number of top sentences to select
        return sorted(sent_options, key=lambda item: ((1-self.fact_penalty)*item.get_avg_log() + self.fact_penalty*item.classification_score), reverse=True)[:n] # sort in descending order
    
    # --------- Generation ---------
    def generate(
            self, 
            input_ids, 
            max_sents: int, 
            num_sents_option: Optional[int] = None, 
            max_length: Optional[int] = None, 
            sent_top_k: Optional[int] = None, 
            num_beams: Optional[int] = None, 
            do_sample: Optional[bool] = None,
            top_p: Optional[float] = None, 
            num_beam_groups: Optional[int] = None, 
            diversity_penalty: Optional[float] = None, 
            length_penalty: Optional[float] = None, 
            **kwargs
        ):
        r"""
        Parameters:
            max_sents: (`int`):
                Number of sentences that will be contained in the result.
            num_sents_option: (`int`, *optional*, defaults to 1):
                Number of sentences that will be generated at each time step.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length of the sequence to be generated.
            sent_top_k (`int`, *optional*, defaults to 1):
                Number of sentences that will be kept at each time step.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            length_penalty (`float`, *optional*, defaults to 1.0):
                 Exponential penalty to the length. 1.0 means that the beam score is penalized by the sequence length.
                 0.0 means no penalty. Set to values < 0.0 in order to encourage the model to generate longer
                 sequences, to a value > 0.0 in order to encourage the model to produce shorter sequences.
        """
        
        do_sample = do_sample if do_sample is not None else self.generator.model.config.do_sample
        num_beams = num_beams if num_beams is not None else self.generator.model.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.generator.model.config.num_beam_groups
        max_length = max_length if max_length is not None else self.generator.model.config.max_length
        
        sent_top_k = sent_top_k if sent_top_k is not None else 1
         
        is_sample_gen_mode = (
            (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        )
        is_beam_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        )
        is_beam_sample_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )
        
        if not is_sample_gen_mode and not is_beam_gen_mode and not is_beam_sample_gen_mode and not is_group_beam_gen_mode:
            raise Exception('Not implemnted method')
        
        output_text = ''
        gen_history = []

        for sent_idx in range(max_sents):
            if is_beam_gen_mode:
                num_beams = num_beams if num_beams is not None else self.generator.model.config.num_beams

                if sent_idx == 0:
                    sent_options = self.generate_sentence_options(
                        sample_size=num_sents_option, 
                        input_ids=input_ids, 
                        num_beams=num_beams, 
                        max_output_length=max_length, 
                        **kwargs)
                    gen_history = self.sort_filter_gen_history(sent_options, sent_top_k) # get top k hypothesis

                else:
                    sent_options = []
                    for _, prev_item in enumerate(gen_history):                       
                        batch_options = self.generate_sentence_options(
                            sample_size=num_sents_option,
                            input_ids=input_ids, 
                            num_beams=num_beams, 
                            top_p=top_p, 
                            max_output_length=max_length,
                            prev_gen=prev_item, 
                            **kwargs)
                        sent_options.extend(batch_options)
        
            elif is_beam_sample_gen_mode:
                num_beams = num_beams if num_beams is not None else self.generator.model.config.num_beams

                if sent_idx == 0:
                    gen_history = self.generate_beamsample_options(
                        sample_size=num_sents_option, 
                        input_ids=input_ids, 
                        num_beams=num_beams, 
                        top_p=top_p, 
                        max_output_length=max_length, 
                        **kwargs)
                else:
                    sent_options = []
                    for _, prev_item in enumerate(gen_history):
                        batch_options = self.generate_beamsample_options(
                            sample_size=num_sents_option, 
                            input_ids=input_ids, 
                            num_beams=num_beams, 
                            top_p=top_p, 
                            max_output_length=max_length, 
                            prev_gen=prev_item, 
                            **kwargs)
                        sent_options.extend(batch_options)

            elif is_sample_gen_mode:
                if sent_idx == 0:
                    gen_history = self.generte_sample_options(
                        sample_size=num_sents_option, 
                        input_ids=input_ids,
                        top_p=top_p,
                        max_output_length=max_length, 
                        **kwargs)
                else:
                    sent_options = []
                    for _, prev_item in enumerate(gen_history):
                        batch_options = self.generte_sample_options(
                            sample_size=num_sents_option, 
                            input_ids=input_ids, 
                            top_p=top_p, 
                            max_output_length=max_length, 
                            prev_gen=prev_item, 
                            **kwargs)
                        sent_options.extend(batch_options)

            elif is_group_beam_gen_mode:
                num_beams = num_beams if num_beams is not None else self.generator.model.config.num_beams
                num_beam_groups = num_beam_groups if num_beam_groups is not None else self.generator.model.config.num_beam_groups

                if sent_idx == 0:
                    gen_history = self.generate_groupbeamsearch_options(
                        sample_size=num_sents_option, 
                        input_ids=input_ids, 
                        num_beams=num_beams, 
                        num_beam_groups=num_beam_groups, 
                        diversity_penalty=diversity_penalty, 
                        length_penalty=length_penalty, 
                        max_output_length=max_length, 
                        **kwargs)
                else:
                    sent_options = []
                    for _, prev_item in enumerate(gen_history):
                        batch_options = self.generate_groupbeamsearch_options(
                            sample_size=num_sents_option, 
                            input_ids=input_ids, 
                            num_beams=num_beams, 
                            num_beam_groups=num_beam_groups, 
                            diversity_penalty=diversity_penalty, 
                            length_penalty=length_penalty, 
                            max_output_length=max_length, 
                            prev_gen=prev_item, 
                            **kwargs)
                        sent_options.extend(batch_options)

            if sent_idx != 0:
                gen_history = self.sort_filter_gen_history(sent_options, sent_top_k) # get top k hypothesis

        if max_sents > 0:
            output_text = self.sort_filter_gen_history(gen_history, 1)[0].text

        return output_text
