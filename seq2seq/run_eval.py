import argparse
import json
from pathlib import Path

import torch

from transformers import AutoTokenizer, BartForConditionalGeneration
from datasets import load_metric

from seq2seq.generation_utils import generate

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

rouge = load_metric("rouge")

def get_rouge_scores(preds, refs):
    rouge_output = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {
        "r1-p": round(rouge_output["rouge1"].mid.precision * 100, 2),
        "r1-r": round(rouge_output["rouge1"].mid.recall * 100, 2),
        "r1-f1": round(rouge_output["rouge1"].mid.fmeasure * 100, 2),
        "r2-p": round(rouge_output["rouge2"].mid.precision * 100, 2),
        "r2-r": round(rouge_output["rouge2"].mid.recall * 100, 2),
        "r2-f1": round(rouge_output["rouge2"].mid.fmeasure * 100, 2),
        "rL-p": round(rouge_output["rougeL"].mid.precision * 100, 2),
        "rL-r": round(rouge_output["rougeL"].mid.recall * 100, 2),
        "rL-f1": round(rouge_output["rougeL"].mid.fmeasure * 100, 2)
    }

def generate_summaries(
    args,
    input_text: str,
    out_file: str,
    model_name: str,
    device: str = DEFAULT_DEVICE,
    **gen_kwargs,
) -> None:
    model_name = str(model_name)
    print(f'Decode with {model_name}')
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(input_text, padding="max_length", max_length=args.max_src_length,
                            return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    eop_token_id = tokenizer('.')[1]

    summaries = generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=args.max_tgt_length,
        num_beams=args.beam_size,
        eop_token_id = eop_token_id,
        tokenizer=tokenizer,
    )
    fout = Path(out_file).open("w", encoding="utf-8")
    dec = tokenizer.batch_decode(summaries[0], skip_special_tokens=True)
    for hypothesis in dec:
        fout.write(hypothesis.strip() + "\n")
        fout.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-base", help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--input_text_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--input_amr_path", type=str, help="where input_text's amr is located")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument("--score_path", type=str, required=False, help="where to save the rouge score in json format")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument('--beam_size', type=int, default=4, help="Beam size for searching")
    parser.add_argument('--max_tgt_length', type=int, default=256, help="maximum length of target sequence")
    parser.add_argument('--max_src_length', type=int, default=1024, help="maximum length of source sequence")
    
    args = parser.parse_args()
    input_text = ' '.join([" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()])

    # add preprocess input

    generate_summaries(
        args,
        input_text,
        args.save_path,
        args.model_name,
        device=args.device,
        decoder_start_token_id=args.decoder_start_token_id,
    )
    if args.reference_path is None:
        return
    
    # Compute scores
    output_lns = [' '.join([x.rstrip() for x in open(args.save_path).readlines()])]
    reference_lns = [' '.join([x.rstrip() for x in open(args.reference_path).readlines()])]
    scores: dict = get_rouge_scores(output_lns, reference_lns)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w+"))
    return scores

if __name__ == "__main__":
    run_generate()