# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils.command.config import config
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation_beam_constraints import Constraint, DisjunctiveConstraint, PhrasalConstraint
from seq2seq.generation_beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)

from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.pytorch_utils import torch_int_div
from transformers.utils import ModelOutput, logging

from seq2seq.faithful_process import FaithfulProcess
from seq2seq.metrics import Metric
from seq2seq.stopping_criteria import MultiBatchEndSentenceCriteria
from seq2seq.sentence_utils import is_sentence_done


logger = logging.get_logger(__name__)

@dataclass
class BeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams*num_return_sequences, config.vocab_size)`.
        beam_indices (`tuple(tuple(torch.LongTensor))`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, input_ids.shape[-1])`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`tuple(tuple(torch.LongTensor))`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, max_length-1)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]

GenerateOutput = BeamSearchOutput

def _prepare_model_inputs(
    model,
    inputs: Optional[torch.Tensor] = None,
    bos_token_id: Optional[int] = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
    """
    This function extracts the model-specific `inputs` for generation.
    """
    # 1. retrieve all kwargs that are non-None or non-model input related.
    # some encoder-decoder models have different names for model and encoder
    if (
        model.config.is_encoder_decoder
        and hasattr(model, "encoder")
        and model.encoder.main_input_name != model.main_input_name
    ):
        input_name = model.encoder.main_input_name
    else:
        input_name = model.main_input_name

    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

    # 2. check whether model_input_name is passed as kwarg
    # if yes and `inputs` is None use kwarg inputs
    inputs_kwarg = model_kwargs.pop(input_name, None)
    if inputs_kwarg is not None and inputs is not None:
        raise ValueError(
            f"`inputs`: {inputs}` were passed alongside "
            f"{input_name} which is not allowed."
            f"Make sure to either pass {inputs} or {input_name}=..."
        )
    elif inputs_kwarg is not None:
        inputs = inputs_kwarg

    # 3. models with `input_ids` can also make use of `inputs_embeds`
    if _can_retrieve_inputs_from_name(model, inputs, "inputs_embeds", model_kwargs):
        inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

    # 4. Only encoder-decoder models can have non `input_ids` input format
    if not model.config.is_encoder_decoder and input_name != "input_ids":
        raise ValueError(
            f"If {input_name} is passed as model-specific keyword "
            "input then model has to be an encoder-decoder and not a "
            f"{model.__class__.__name__}."
        )

    # 5. if `inputs` is still None, try to create `input_ids` from BOS token
    if inputs is None:
        inputs = _prepare_input_ids_for_generation(model, bos_token_id, model_kwargs.get("encoder_outputs"))

    return inputs, input_name, model_kwargs

def _can_retrieve_inputs_from_name(
    model, inputs: Optional[torch.Tensor], name: str, model_kwargs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    If `inputs` is None and `name` is in both forward function and keyword arguments, then inputs can be retrieved
    from name
    """
    can_retrieve_inputs = model_kwargs.get(name, None) is not None and name in set(
        inspect.signature(model.forward).parameters.keys()
    )

    if can_retrieve_inputs and inputs is not None:
        raise ValueError(f"Cannot only pass one of {name} and {model.main_input_name}")

    return can_retrieve_inputs

def _prepare_input_ids_for_generation(
    model, bos_token_id: Optional[int], encoder_outputs: Optional[ModelOutput]
) -> torch.LongTensor:
    if model.config.is_encoder_decoder and encoder_outputs is not None:
        # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
        shape = encoder_outputs.last_hidden_state.size()[:-1]
        return torch.ones(shape, dtype=torch.long, device=model.device) * -100

    if bos_token_id is None:
        raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")
    return torch.ones((1, 1), dtype=torch.long, device=model.device) * bos_token_id

def _prepare_attention_mask_for_generation(
    inputs: torch.Tensor,
    pad_token_id: Optional[int],
    eos_token_id: Optional[int],
) -> torch.LongTensor:
    is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
    is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
    is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)

    # Check if input is input_ids and padded -> only then is attention_mask defined
    if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
        return inputs.ne(pad_token_id).long()
    else:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

def _prepare_encoder_decoder_kwargs_for_generation(
    model, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
) -> Dict[str, Any]:
    # 1. get encoder
    encoder = model.get_encoder()

    # 2. prepare encoder args and encoder kwargs from model kwargs
    irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
    encoder_kwargs = {
        argument: value
        for argument, value in model_kwargs.items()
        if not any(argument.startswith(p) for p in irrelevant_prefix)
    }

    # 3. make sure that encoder returns `ModelOutput`
    model_input_name = model_input_name if model_input_name is not None else model.main_input_name
    encoder_kwargs["return_dict"] = True
    encoder_kwargs[model_input_name] = inputs_tensor
    model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

    return model_kwargs

def _prepare_decoder_input_ids_for_generation(
    model,
    batch_size: int,
    decoder_start_token_id: int = None,
    bos_token_id: int = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    device: torch.device = None,
) -> torch.LongTensor:
    if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
        return model_kwargs.pop("decoder_input_ids")
    else:
        decoder_start_token_id = _get_decoder_start_token_id(model, decoder_start_token_id, bos_token_id)
        if device is None:
            device = model.device
        return torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id

def _get_decoder_start_token_id(model, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else model.config.decoder_start_token_id
    )
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id

    if decoder_start_token_id is not None:
        return decoder_start_token_id
    elif (
        hasattr(model.config, "decoder")
        and hasattr(model.config.decoder, "decoder_start_token_id")
        and model.config.decoder.decoder_start_token_id is not None
    ):
        return model.config.decoder.decoder_start_token_id
    elif bos_token_id is not None:
        return bos_token_id
    elif (
        hasattr(model.config, "decoder")
        and hasattr(model.config.decoder, "bos_token_id")
        and model.config.decoder.bos_token_id is not None
    ):
        return model.config.decoder.bos_token_id
    raise ValueError(
        "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
    )

def _expand_inputs_for_generation(
    input_ids: torch.LongTensor,
    expand_size: int = 1,
    is_encoder_decoder: bool = False,
    attention_mask: Optional[torch.LongTensor] = None,
    encoder_outputs: Optional[ModelOutput] = None,
    **model_kwargs,
) -> Tuple[torch.LongTensor, Dict[str, Any]]:
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

    if is_encoder_decoder:
        if encoder_outputs is None:
            raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )
        model_kwargs["encoder_outputs"] = encoder_outputs
    return input_ids, model_kwargs

def _update_model_kwargs_for_generation(
    outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
) -> Dict[str, Any]:
    # update past
    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs.past_key_values
    elif "mems" in outputs:
        model_kwargs["past"] = outputs.mems
    elif "past_buckets_states" in outputs:
        model_kwargs["past"] = outputs.past_buckets_states
    else:
        model_kwargs["past"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if not is_encoder_decoder:
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return model_kwargs

def _get_logits_warper(
    model,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    typical_p: Optional[float] = None,
    temperature: Optional[float] = None,
    num_beams: Optional[int] = None,
    renormalize_logits: Optional[bool] = None,
) -> LogitsProcessorList:
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
    used for multinomial sampling.
    """

    # init warp parameters
    top_k = top_k if top_k is not None else model.config.top_k
    top_p = top_p if top_p is not None else model.config.top_p
    typical_p = typical_p if typical_p is not None else model.config.typical_p
    temperature = temperature if temperature is not None else model.config.temperature
    # instantiate warpers list
    warpers = LogitsProcessorList()

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    if typical_p is not None and typical_p < 1.0:
        warpers.append(TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    # `LogitNormalization` should always be the last logit processor, when present
    if renormalize_logits is True:
        warpers.append(LogitNormalization())
    return warpers

def _get_logits_processor(
    model,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    encoder_no_repeat_ngram_size: int,
    input_ids_seq_length: int,
    encoder_input_ids: torch.LongTensor,
    bad_words_ids: List[List[int]],
    min_length: int,
    max_length: int,
    eos_token_id: int,
    forced_bos_token_id: int,
    forced_eos_token_id: int,
    prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    num_beams: int,
    num_beam_groups: int,
    diversity_penalty: float,
    remove_invalid_values: bool,
    exponential_decay_length_penalty: Tuple,
    logits_processor: Optional[LogitsProcessorList],
    renormalize_logits: Optional[bool],
    suppress_tokens: Optional[List[int]] = None,
    begin_suppress_tokens: Optional[List[int]] = None,
    forced_decoder_ids: Optional[List[List[int]]] = None,
) -> LogitsProcessorList:
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
    instances used to modify the scores of the language model head.
    """
    processors = LogitsProcessorList()

    # init warp parameters
    repetition_penalty = repetition_penalty if repetition_penalty is not None else model.config.repetition_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else model.config.no_repeat_ngram_size
    )
    encoder_no_repeat_ngram_size = (
        encoder_no_repeat_ngram_size
        if encoder_no_repeat_ngram_size is not None
        else model.config.encoder_no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else model.config.bad_words_ids
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    diversity_penalty = diversity_penalty if diversity_penalty is not None else model.config.diversity_penalty
    forced_bos_token_id = (
        forced_bos_token_id if forced_bos_token_id is not None else model.config.forced_bos_token_id
    )
    forced_eos_token_id = (
        forced_eos_token_id if forced_eos_token_id is not None else model.config.forced_eos_token_id
    )
    remove_invalid_values = (
        remove_invalid_values if remove_invalid_values is not None else model.config.remove_invalid_values
    )
    exponential_decay_length_penalty = (
        exponential_decay_length_penalty
        if exponential_decay_length_penalty is not None
        else model.config.exponential_decay_length_penalty
    )
    suppress_tokens = suppress_tokens if suppress_tokens is not None else model.config.suppress_tokens
    begin_suppress_tokens = (
        begin_suppress_tokens if begin_suppress_tokens is not None else model.config.begin_suppress_tokens
    )
    if forced_decoder_ids is None and hasattr(model.config, "forced_decoder_ids"):
        forced_decoder_ids = model.config.forced_decoder_ids
    # instantiate processors list

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
        if model.config.is_encoder_decoder:
            processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
        else:
            raise ValueError(
                "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
            )
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > 0:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    if exponential_decay_length_penalty is not None:
        processors.append(
            ExponentialDecayLengthPenalty(exponential_decay_length_penalty, eos_token_id, input_ids_seq_length)
        )
    if suppress_tokens is not None:
        processors.append(SuppressTokensLogitsProcessor(suppress_tokens))
    if begin_suppress_tokens is not None:
        begin_index = input_ids_seq_length
        begin_index = begin_index if (input_ids_seq_length > 1 or forced_bos_token_id is None) else begin_index + 1
        if forced_decoder_ids is not None:
            begin_index += forced_decoder_ids[-1][0]  # generation starts after the last token that is forced
        processors.append(SuppressTokensAtBeginLogitsProcessor(begin_suppress_tokens, begin_index))
    if forced_decoder_ids is not None:
        processors.append(ForceTokensLogitsProcessor(forced_decoder_ids))
    processors = _merge_criteria_processor_list(processors, logits_processor)
    # `LogitNormalization` should always be the last logit processor, when present
    if renormalize_logits is True:
        processors.append(LogitNormalization())
    return processors

def _get_stopping_criteria(
    max_length: Optional[int], max_time: Optional[float], stopping_criteria: Optional[StoppingCriteriaList],
) -> StoppingCriteriaList:
    criteria = StoppingCriteriaList()
    if max_length is not None:
        criteria.append(MaxLengthCriteria(max_length=max_length))
    if max_time is not None:
        criteria.append(MaxTimeCriteria(max_time=max_time))
    criteria = _merge_criteria_processor_list(criteria, stopping_criteria)
    return criteria

def _merge_criteria_processor_list(
    default_list: Union[LogitsProcessorList, StoppingCriteriaList],
    custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
) -> Union[LogitsProcessorList, StoppingCriteriaList]:
    if len(custom_list) == 0:
        return default_list
    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                raise ValueError(
                    f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                    f" `generate`, but it has already been created with the values {default}. {default} has been"
                    " created by passing the corresponding arguments to generate or by the model's config default"
                    f" values. If you just want to change the default values of {object_type} consider passing"
                    f" them as arguments to `generate` instead of using a custom {object_type}."
                )
    default_list.extend(custom_list)
    return default_list

def compute_transition_beam_scores(
    model,
    sequences: torch.Tensor,
    scores: Tuple[torch.Tensor],
    beam_indices: torch.Tensor,
    eos_token_id: int = None,
):
    """compute the transition probabilities of sequences given generation
    scores and beam indices"""

    # 1. reshape scores as [vocab_size * batch_size, # generation steps]
    # with batch_size being 2 * vocab_size and # generation steps being
    # seq_len - input_length
    scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

    # 2. cut beam_indices to longest beam length
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    beam_indices = beam_indices[:, :max_beam_length]
    beam_indices_mask = beam_indices_mask[:, :max_beam_length]

    # 3. Set indices of beams that finished early to 0
    # such indices will be masked correctly afterwards
    beam_indices[beam_indices_mask] = 0

    # 4. multiply beam_indices with vocab size to gather correctly from scores
    beam_sequence_indices = beam_indices * model.config.vocab_size

    # 5. Define which indices contributed to scores
    cut_idx = sequences.shape[-1] - max_beam_length
    indices = sequences[:, cut_idx:] + beam_sequence_indices

    # 6. Compute scores
    transition_scores = scores.gather(0, indices)

    # 7. Mask out transition_scores of beams that stopped early
    transition_scores[beam_indices_mask] = 0

    return transition_scores


@torch.no_grad()
def generate(
    model,
    inputs: Optional[torch.Tensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    penalty_alpha: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    typical_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    encoder_no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    max_time: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    num_beam_groups: Optional[int] = None,
    diversity_penalty: Optional[float] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    renormalize_logits: Optional[bool] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    constraints: Optional[List[Constraint]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    forced_bos_token_id: Optional[int] = None,
    forced_eos_token_id: Optional[int] = None,
    remove_invalid_values: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    exponential_decay_length_penalty: Optional[Tuple[int, float]] = None,
    suppress_tokens: Optional[List[int]] = None,
    begin_suppress_tokens: Optional[List[int]] = None,
    forced_decoder_ids: Optional[List[List[int]]] = None,
    **model_kwargs,
)->Union[GenerateOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head. The method supports the following
    generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:
        - *greedy decoding* by calling [`~generation_utils.GenerationMixin.greedy_search`] if `num_beams=1` and
            `do_sample=False`.
        - *contrastive search* by calling [`~generation_utils.GenerationMixin.contrastive_search`] if
            `penalty_alpha>0.` and `top_k>1`
        - *multinomial sampling* by calling [`~generation_utils.GenerationMixin.sample`] if `num_beams=1` and
            `do_sample=True`.
        - *beam-search decoding* by calling [`~generation_utils.GenerationMixin.beam_search`] if `num_beams>1` and
            `do_sample=False`.
        - *beam-search multinomial sampling* by calling [`~generation_utils.GenerationMixin.beam_sample`] if
            `num_beams>1` and `do_sample=True`.
        - *diverse beam-search decoding* by calling [`~generation_utils.GenerationMixin.group_beam_search`], if
            `num_beams>1` and `num_beam_groups>1`.
        - *constrained beam-search decoding* by calling
            [`~generation_utils.GenerationMixin.constrained_beam_search`], if `constraints!=None` or
            `force_words_ids!=None`.
    <Tip warning={true}>
    Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name as
    defined in the model's config (`config.json`) which in turn defaults to the
    [`~modeling_utils.PretrainedConfig`] of the model.
    </Tip>
    Most of these parameters are explained in more detail in [this blog
    post](https://huggingface.co/blog/how-to-generate).
    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        max_length (`int`, *optional*, defaults to `model.config.max_length`):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in
            the prompt.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        min_length (`int`, *optional*, defaults to `model.config.min_length` or 10 if the config does not set any value):
            The minimum length of the sequence to be generated.
        do_sample (`bool`, *optional*, defaults to `model.config.do_sample` or `False` if the config does not set any value):
            Whether or not to use sampling ; use greedy decoding otherwise.
        early_stopping (`bool`, *optional*, defaults to `False`):
            Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
        num_beams (`int`, *optional*, defaults to `model.config.num_beams` or 1 if the config does not set any value):
            Number of beams for beam search. 1 means no beam search.
        temperature (`float`, *optional*, defaults to `model.config.temperature` or 1.0 if the config does not set any value):
            The value used to module the next token probabilities.
        penalty_alpha (`float`, *optional*, defaults to `model.config.penalty_alpha` or None if the config does not set any value):
            The values balance the model confidence and the degeneration penalty in contrastive search decoding.
        top_k (`int`, *optional*, defaults to `model.config.top_k` or 50 if the config does not set any value):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to `model.config.top_p` or 1.0 if the config does not set any value):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        typical_p (`float`, *optional*, defaults to `model.config.typical_p` or 1.0 if the config does not set any value):
            The amount of probability mass from the original distribution to be considered in typical decoding. If
            set to 1.0 it takes no effect. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
        repetition_penalty (`float`, *optional*, defaults to `model.config.repetition_penalty` or 1.0 if the config does not set any value):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        pad_token_id (`int`, *optional*, defaults to `model.config.pad_token_id`):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to `model.config.bos_token_id`):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to `model.config.eos_token_id`):
            The id of the *end-of-sequence* token.
        length_penalty (`float`, *optional*, defaults to `model.config.length_penalty` or 1.0 if the config does not set any value):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent
            to the sequence length, which in turn is used to divide the score of the sequence. Since the score is
            the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences,
            while `length_penalty` < 0.0 encourages shorter sequences.
        no_repeat_ngram_size (`int`, *optional*, defaults to `model.config.no_repeat_ngram_size` or 0 if the config does not set any value):
            If set to int > 0, all ngrams of that size can only occur once.
        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to `model.config.encoder_no_repeat_ngram_size` or 0 if the config does not set any value):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        bad_words_ids(`List[List[int]]`, *optional*, defaults to `model.config.bad_words_ids`):
            List of token ids that are not allowed to be generated. In order to get the token ids of the words that
            should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        force_words_ids(`List[List[int]]` or `List[List[List[int]]]`, *optional*):
            List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple
            list of words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`,
            this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081),
            where one can allow different forms of each word.
        num_return_sequences(`int`, *optional*, defaults to `model.config.num_return_sequences` or 1 if the config does not set any value):
            The number of independently computed returned sequences for each element in the batch.
        max_time(`float`, *optional*):
            The maximum amount of time you allow the computation to run for in seconds. generation will still
            finish the current pass after allocated time has been passed.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
            that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same shape
            as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
        decoder_start_token_id (`int`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        num_beam_groups (`int`, *optional*, defaults to `model.config.num_beam_groups` or 1 if the config does not set any value):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
            beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        diversity_penalty (`float`, *optional*, defaults to `model.config.diversity_penalty` or 0.0 if the config does not set any value):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group
            at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
            enabled.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and a
                model's config. If a logit processor is passed that is already created with the arguments or a model's
                config an error is thrown. This feature is intended for advanced users.
        renormalize_logits (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the logits after applying all the logits processors or warpers (including the
            custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the
            score logits are normalized but some logit processors or warpers break the normalization.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                model's config. If a stopping criteria is passed that is already created with the arguments or a
                model's config an error is thrown. This feature is intended for advanced users.
        constraints (`List[Constraint]`, *optional*):
                Custom constraints that can be added to the generation to ensure that the output will contain the use
                of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
        output_attentions (`bool`, *optional*, defaults to `model.config.output_attentions` or `False` if the config does not set any value):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `model.config.output_hidden_states` or `False` if the config does not set any value):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `model.config.output_scores` or `False` if the config does not set any value):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `model.config.return_dict_in_generate` or `False` if the config does not set any value):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
            for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
            the target language token.
        forced_eos_token_id (`int`, *optional*, defaults to `model.config.forced_eos_token_id`):
            The id of the token to force as the last generated token when `max_length` is reached.
        remove_invalid_values (`bool`, *optional*, defaults to `model.config.remove_invalid_values`):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
            crash. Note that using `remove_invalid_values` can slow down generation.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*, defaults to `model.config.exponential_decay_length_penalty`):
            This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
            generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates
            where penalty starts and `decay_factor` represents the factor of exponential decay
        suppress_tokens  (`List[int]`, *optional*, defaults to `model.config.suppress_tokens`):
            A list of tokens that will be supressed at generation. The `SupressTokens` logit processor will set
            their log probs to `-inf` so that they are not sampled.
        begin_suppress_tokens  (`List[int]`, *optional*, defaults to `model.config.begin_suppress_tokens`):
            A list of tokens that will be supressed at the begining of the generation. The `SupressBeginTokens`
            logit processor will set their log probs to `-inf` so that they are not sampled.
        forced_decoder_ids (`List[List[int]]`, *optional*, defaults to `model.config.forced_decoder_ids`):
            A list of pairs of integers which indicates a mapping from generation indices to token indices that
            will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always
            be a token of index 123.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
            is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
            should be prefixed with *decoder_*.

    Return:
        `torch.LongTensor`

    Examples:

    Beam-search decoding:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        >>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        >>> sentence = "Paris is one of the densest populated areas in Europe."
        >>> input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        >>> outputs = model.generate(input_ids, num_beams=5)
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Paris ist eines der dichtesten besiedelten Gebiete Europas.']
        ```
    """

    # 1. Set generation parameters if not already defined
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    num_beams = num_beams if num_beams is not None \
        else model.config.num_beams if hasattr(model.config, 'numn_beams') and model.config.num_beams is not None \
            else 1
    length_penalty = length_penalty if length_penalty is not None \
        else model.config.length_penalty if hasattr(model.config, 'length_penalty') and model.config.length_penalty is not None \
            else 1.0
    early_stopping = early_stopping if early_stopping is not None \
        else model.config.early_stopping if hasattr(model.config, 'early_stopping') and model.config.early_stopping is not None \
            else False
    num_beam_groups = num_beam_groups if num_beam_groups is not None \
        else model.config.num_beam_groups if hasattr(model.config, 'num_beam_groups') and model.config.num_beam_groups is not None \
            else 1
    do_sample = do_sample if do_sample is not None \
        else model.config.do_sample if hasattr(model.config, 'do_sample') and model.config.do_sample is not None \
            else False
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None 
        else model.config.num_return_sequences if hasattr(model.config, 'num_return_sequences') and model.config.num_return_sequences is not None
        else 1
    )
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id

    if eos_token_id is None and hasattr(model.config, "decoder"):
        eos_token_id = model.config.decoder.eos_token_id

    if pad_token_id is None and eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    output_scores = output_scores if output_scores is not None \
        else model.config.output_scores if hasattr(model.config, 'output_scores') and model.config.output_scores is not None \
            else False
    output_attentions = output_attentions if output_attentions is not None \
        else model.config.output_attentions if hasattr(model.config, 'output_attentions') and model.config.output_attentions is not None \
            else False
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None 
        else model.config.output_hidden_states if hasattr(model.config, 'output_hidden_states') and model.config.output_hidden_states is not None 
        else False
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None
        else model.config.return_dict_in_generate if hasattr(model.config, 'return_dict_in_generate') and model.config.return_dict_in_generate is not None
        else False
    )

    ##########################################################
    tokenizer = model_kwargs.pop("tokenizer", None)
    amrs_input_path:Optional[str] = model_kwargs.pop("amrs_input_path", None)
    metric:Optional[Metric] = model_kwargs.pop("metric", None)
    faithful_penalty:Optional[float] = model_kwargs.pop("faithful_penalty", None)
    num_sentences_before_penalty: Optional[int] = model_kwargs.pop("num_sentences_before_penalty", None)

    # 2. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = _prepare_model_inputs(model, inputs, bos_token_id, model_kwargs)
    batch_size = inputs_tensor.shape[0]

    # 3. Define other model kwargs
    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states
    model_kwargs["use_cache"] = use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = _prepare_attention_mask_for_generation(
            inputs_tensor, pad_token_id, eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not model.config.is_encoder_decoder:
        if pad_token_id is not None and torch.sum(inputs_tensor[:, -1] == pad_token_id) > 0:
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = _prepare_encoder_decoder_kwargs_for_generation(
            model, inputs_tensor, model_kwargs, model_input_name
        )

    # 4. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids = _prepare_decoder_input_ids_for_generation(
            model,
            batch_size,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,
            model_kwargs=model_kwargs,
            device=inputs_tensor.device,
        )
    else:
        # if decoder-only then inputs_tensor has to be `input_ids`
        input_ids = inputs_tensor

    # 5. Prepare `max_length` depending on other stopping criteria.
    input_ids_seq_length = input_ids.shape[-1]
    if max_length is None and max_new_tokens is None:
        warnings.warn(
            "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to "
            f"{model.config.max_length} (`model.config.max_length`). Controlling `max_length` via the config is "
            "deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend "
            "using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif max_length is None and max_new_tokens is not None:
        max_length = max_new_tokens + input_ids_seq_length
    elif max_length is not None and max_new_tokens is not None:
        raise ValueError(
            "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
            " limit to the generated output length. Remove one of those arguments. Please refer to the"
            " documentation for more information. "
            "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
        )
    # default to config if still None
    max_length = max_length if max_length is not None else model.config.max_length
    min_length = min_length if min_length is not None else model.config.min_length if hasattr(model.config, 'min_length') else 10

    if min_length is not None and min_length > max_length:
        raise ValueError(
            f"Unfeasible length constraints: the minimum length ({min_length}) is larger than the maximum "
            f"length ({max_length})"
        )
    if input_ids_seq_length >= max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {max_length}. This can lead to unexpected behavior. You should consider increasing "
            "`max_new_tokens`."
        )

    # 6. determine generation mode
    is_constraint_gen_mode = constraints is not None or force_words_ids is not None

    is_contrastive_search_gen_mode = (
        top_k is not None and top_k > 1 and do_sample is False and penalty_alpha is not None and penalty_alpha > 0
    )

    is_greedy_gen_mode = (
        (num_beams == 1)
        and (num_beam_groups == 1)
        and do_sample is False
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_sample_gen_mode = (
        (num_beams == 1)
        and (num_beam_groups == 1)
        and do_sample is True
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_beam_gen_mode = (
        (num_beams > 1)
        and (num_beam_groups == 1)
        and do_sample is False
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_beam_sample_gen_mode = (
        (num_beams > 1)
        and (num_beam_groups == 1)
        and do_sample is True
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )
    is_group_beam_gen_mode = (
        (num_beams > 1)
        and (num_beam_groups > 1)
        and not is_constraint_gen_mode
        and not is_contrastive_search_gen_mode
    )

    if num_beam_groups > num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    if model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 7. prepare distribution pre_processing samplers
    logits_processor = _get_logits_processor(
        model=model,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        remove_invalid_values=remove_invalid_values,
        exponential_decay_length_penalty=exponential_decay_length_penalty,
        logits_processor=logits_processor,
        renormalize_logits=renormalize_logits,
        suppress_tokens=suppress_tokens,
        begin_suppress_tokens=begin_suppress_tokens,
        forced_decoder_ids=forced_decoder_ids,
    )

    faithful_process = FaithfulProcess(
        tokenizer=tokenizer,
        amrs_input_path_name=amrs_input_path, 
        metric=metric,
        faithful_penalty=faithful_penalty,
    ) if (tokenizer is not None) and (amrs_input_path is not None) and (faithful_penalty is not None and faithful_penalty > 0) else None

    # 8. prepare stopping criteria
    stopping_criteria = _get_stopping_criteria(
        max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria,
    )
    # 9. go into different generation modes
    if is_beam_gen_mode:
        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        # 10. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = _expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=model.config.is_encoder_decoder, **model_kwargs
        )

        stopping_criteria.append(MultiBatchEndSentenceCriteria(pad_token_id))

        past_scores = None
        while input_ids.shape[-1] < max_length:

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_beams,
                tokenizer=tokenizer,
            )

            # 12. run beam search
            bs = beam_search(
                model,
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                synced_gpus=synced_gpus,
                past_scores=past_scores,
                **model_kwargs,
            )
            # process new sentence
            old_sequence_len = input_ids.shape[-1]
            input_ids = bs['sequences']
            past_scores = bs['sequences_scores']

            # apply faithful penalty
            if faithful_process is not None:
                past_scores = faithful_process(input_ids[:, old_sequence_len:], past_scores)

            print('*'*100)
            print('sentences complete\n')
            for idx in input_ids:
                print(tokenizer.decode(idx))
            print('*'*100)

            predicted_sentences = [[sequence_ids, score] for sequence_ids, score in zip(input_ids.clone(), past_scores.clone())]
            sorted_predicted_sentences = sorted(predicted_sentences, key=lambda x: x[1], reverse=True)
            best_hyp = sorted_predicted_sentences[0]

            # check if the eos of best hypotesis came from decoding step or if it has been added in finalize step
            if input_ids[:-1].shape[-1] < max_length - 1:
                # cut off pad and eos ids
                pad_mask = best_hyp[0]==pad_token_id
                eos_mask = best_hyp[0]==eos_token_id
                comb_mask = pad_mask.logical_or(eos_mask)
                last_valid_idx = (comb_mask == False).nonzero()[-1][-1].item() 
                curr_gen_ids = best_hyp[0][:last_valid_idx+1]

                # replace all beam sequence with the best sequence
                input_ids = curr_gen_ids.repeat(num_beams, 1)
                continue

            input_ids = input_ids.new([sent[0].tolist() for sent in sorted_predicted_sentences])
            past_scores = past_scores.new([sent[1].item() for sent in sorted_predicted_sentences])

        if return_dict_in_generate:
            if not output_scores:
                bs["sequence_scores"] = None
            num_return_sequences_scores = []
            for beam in bs["scores"]:
                num_return_sequences_scores.append(beam[:num_return_sequences])
            return {
                "sequences": input_ids[:num_return_sequences],
                "sequences_scores": past_scores[:num_return_sequences],
                "scores": tuple(num_return_sequences_scores),
                "beam_indices": bs["beam_indices"][:num_return_sequences],
            }
        else:
            return input_ids[:num_return_sequences]

    elif is_group_beam_gen_mode:
        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        if num_beams % num_beam_groups != 0:
            raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        if typical_p is not None:
            raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

        # 10. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = _expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=model.config.is_encoder_decoder, **model_kwargs
        )

        stopping_criteria.append(MultiBatchEndSentenceCriteria(pad_token_id))

        past_scores = None
        while input_ids.shape[-1] < max_length:

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_beams,
                num_beam_groups=num_beam_groups,
                tokenizer=tokenizer,
            )

            # 12. run beam search
            diverse_bs = group_beam_search(
                model,
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                synced_gpus=synced_gpus,
                past_scores=past_scores,
                **model_kwargs,
            )
            
            old_sequence_len = input_ids.shape[-1]
            input_ids = diverse_bs['sequences']
            past_scores = diverse_bs['sequences_scores']

            # apply faithful penalty
            if faithful_process is not None:
                past_scores = faithful_process(input_ids[:, old_sequence_len:], past_scores)

            print('*'*100)
            print('sentences complete\n')
            for idx in input_ids:
                print(tokenizer.decode(idx))
            print('*'*100)

            predicted_sentences = [[sequence_ids, score] for sequence_ids, score in zip(input_ids.clone(), past_scores.clone())]
            sorted_predicted_sentences = sorted(predicted_sentences, key=lambda x: x[1], reverse=True)
            best_hyp = sorted_predicted_sentences[0]

            # check if the eos of best hypotesis came from decoding step or if it has been added in finalize step
            if input_ids[:-1].shape[-1] < max_length - 1:
                # cut off pad and eos ids
                pad_mask = best_hyp[0]==pad_token_id
                eos_mask = best_hyp[0]==eos_token_id
                comb_mask = pad_mask.logical_or(eos_mask)
                last_valid_idx = (comb_mask == False).nonzero()[-1][-1].item() 
                curr_gen_ids = best_hyp[0][:last_valid_idx+1]

                # replace all beam sequence with the best sequence
                input_ids = curr_gen_ids.repeat(num_beams, 1)
                continue

            input_ids = input_ids.new([sent[0].tolist() for sent in sorted_predicted_sentences])
            past_scores = past_scores.new([sent[1].item() for sent in sorted_predicted_sentences])

        if return_dict_in_generate:
            if not output_scores:
                diverse_bs["sequence_scores"] = None
            num_return_sequences_scores = []
            for beam in diverse_bs["scores"]:
                num_return_sequences_scores.append(beam[:num_return_sequences])
            return {
                "sequences": input_ids[:num_return_sequences],
                "sequences_scores": past_scores[:num_return_sequences],
                "scores": tuple(num_return_sequences_scores),
                "beam_indices": diverse_bs["beam_indices"][:num_return_sequences],
            }
        else:
            return input_ids[:num_return_sequences]

    else:
        raise NotImplementedError

def beam_search(
    model,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.
    
    Return:
        [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    
    Examples:
    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id
    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }
    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )
    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )
    >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    # init values
    # list of penalty objects
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    # list of stopping criteria objects
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    
    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape
    ##################################
    out_cur_len = cur_len
    past_scores = model_kwargs.pop("past_scores", None)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    ##############################################################
    if past_scores is None:
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
    else:
        beam_scores = past_scores
        beam_scores[1:] = -1e9

    # SentBS: flag to detect if a new sentence is produced
    prev_sent_end = True # avoid take previous sentence as new generated sentence
    
    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # input_ids in model_inputs (used for computing probability distribution)
        # is set to None in case of encoder-decoder architectures (e.g., BART), where the input considered is the encoder output (and not the tokenized prompt)
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # vocabulary distribution probability for each beam with respect to the batch size
        # E.g. [[num_beam, batch_size, vocab_size]]
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # store score before decoding step
        scores_before_decoding = beam_scores.clone()

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        # vocabulary distribution probability for each beam
        next_token_logits = outputs.logits[:, -1, :]
        # ensure that the first and the last tokens are BOS and EOS, respectively
        next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        # normalize distribution probability for each beam
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        # refine probability scores according to the eventually specified penalties
        # Note: in case of more penalties, the score is refined sequentially (pipeline)
        # Note: penalty factor can transform some scores to -inf
        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        # beam_scores[:, None] = reshape of beam scores
        # expands_as: for each beam, it replicates the score (0 for the first, 1e-9 for the others) many times as the vocab size
        # E.g. torch.Size([4, 50265]) shape after expand
        # tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
        #          [0., -inf, -inf,  ..., -inf, -inf, -inf],
        #          [0., -inf, -inf,  ..., -inf, -inf, -inf],
        #          [0., -inf, -inf,  ..., -inf, -inf, -inf]], device='cuda:0')
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        # [batch_size, num_beams * vocab_size]; E.g. torch.Size([1, 4*50265])
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 2 new tokens for each beam (so we have some spare tokens and match output of beam search)
        # E.g. next_token_scores: tensor([[ 0.0000e+00, -1.0000e+09, -1.0000e+09, -1.0000e+09,        -inf,
        #                        -inf,        -inf,        -inf]], device='cuda:0')
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = torch_int_div(next_tokens, vocab_size)
        # only after this line, the next_tokens contains a list of real vocabulary-indexed tokens
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            out_cur_len=out_cur_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)


        model_kwargs = _update_model_kwargs_for_generation(
            outputs, model_kwargs=model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if prev_sent_end:
            prev_sent_end = False
        else:
            # SentBS: if previously sentence has ended, change new token to pad_token_id
            input_ids = beam_scorer.is_sentences_complete(
                input_ids, 
                scores_before_decoding, 
                beam_indices=beam_indices, 
                pad_token_id=pad_token_id,
                out_cur_len=out_cur_len,
            )

        # beam search end condition
        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if model.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]

def group_beam_search(
    model,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    **model_kwargs,
):
    r"""
    Generates sequences of token ids for models with a language modeling head using **diverse beam search
    decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
            model is an encoder-decoder model the kwargs should include `encoder_outputs`.
    Return:
        [`~generation_utils.BeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.BeamSearchDecoderOnlyOutput`] if [`~generation_utils.BeamSearchDecoderOnlyOutput`] if
        `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
        [`~generation_utils.BeamSearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.
    Examples:
    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     HammingDiversityLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch
    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
    >>> # lets run diverse beam search using 6 beams
    >>> num_beams = 6
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id
    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }
    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     max_length=model.config.max_length,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ...     num_beam_groups=3,
    ... )
    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )
    >>> outputs = model.group_beam_search(
    ...     input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
    ... )
    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    output_scores = output_scores if output_scores is not None else model.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else model.config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams
    num_beam_groups = beam_scorer.num_beam_groups
    num_sub_beams = num_beams // num_beam_groups
    device = input_ids.device

    batch_beam_size, cur_len = input_ids.shape
    ##################################
    out_cur_len = cur_len
    past_scores = model_kwargs.pop("past_scores", None)

    if return_dict_in_generate and output_scores:
        beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
    else:
        beam_indices = None

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and model.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
    # the same group don't produce same tokens everytime.
    if past_scores is None:
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))
    else:
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = past_scores[0]
        beam_scores = beam_scores.view((batch_size * num_beams,))

    # SentBS: flag to detect if a new sentence is produced
    prev_sent_end = True # avoid take previous sentence as new generated sentence

    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # predicted tokens in cur_len step
        current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

        # indices which will form the beams in the next time step
        reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

        # do one decoder step on all beams of all sentences in batch
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # store score before decoding step
        scores_before_decoding = beam_scores.clone()

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        if output_scores:
            processed_score = torch.zeros_like(outputs.logits[:, -1, :])

        for beam_group_idx in range(num_beam_groups):
            group_start_idx = beam_group_idx * num_sub_beams
            group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
            group_size = group_end_idx - group_start_idx

            # indices of beams of current group among all sentences in batch
            batch_group_indices = []

            for batch_idx in range(batch_size):
                batch_group_indices.extend(
                    [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                )
            group_input_ids = input_ids[batch_group_indices]

            # select outputs of beams of current group only
            next_token_logits = outputs.logits[batch_group_indices, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * group_size, vocab_size)
            vocab_size = next_token_scores.shape[-1]

            next_token_scores_processed = logits_processor(
                group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
            )
            next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
            next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

            if output_scores:
                processed_score[batch_group_indices] = next_token_scores_processed

            # reshape for beam search
            next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True     
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
            beam_outputs = beam_scorer.process(
                group_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=process_beam_indices,
                out_cur_len=out_cur_len,
            )

            beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            if return_dict_in_generate and output_scores:
                beam_indices[beam_group_idx] = tuple(
                    beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
                )

            input_ids[batch_group_indices] = group_input_ids[beam_idx]
            group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            current_tokens[batch_group_indices] = group_input_ids[:, -1]
            
            # (beam_idx // group_size) -> batch_idx
            # (beam_idx % group_size) -> offset of idx inside the group
            reordering_indices[batch_group_indices] = (
                num_beams * torch_int_div(beam_idx, group_size) + group_start_idx + (beam_idx % group_size)
            )

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (processed_score,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                )
                if model.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if model.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)
        
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = model._reorder_cache(model_kwargs["past"], reordering_indices)

        # increase cur_len
        cur_len = cur_len + 1
        
        if prev_sent_end:
            prev_sent_end = False
        else:
            # SentBS: if previously sentence has ended, change new token to pad_token_id
            input_ids = beam_scorer.is_sentences_complete(
                input_ids, 
                scores_before_decoding, 
                beam_indices=beam_indices, 
                pad_token_id=pad_token_id,
                out_cur_len=out_cur_len,
            )

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=final_beam_indices,
        out_cur_len=out_cur_len,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if model.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]
