from pathlib import Path
import argparse,json,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from experiment import load_jsonl,run_orientation
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--split',required=True); p.add_argument('--index',type=int,required=True); p.add_argument('--orientation',required=True); args=p.parse_args(); torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True); model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True); model.eval(); model.requires_grad_(False); task=load_jsonl(ROOT/'data'/f'{args.split}.jsonl')[args.index]; row=run_orientation(model,tok,task,14,args.orientation); dest=ROOT/'results'/args.split/f'{args.index:02d}_{args.orientation}.json'; dest.write_text(json.dumps(row,indent=2)+'\n'); print(dest)
