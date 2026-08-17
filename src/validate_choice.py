from pathlib import Path
import argparse,json,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from fast_suffix import build_fast_cache
from experiment import base_text,pair_batch,walk,load_jsonl
ROOT=Path(__file__).resolve().parents[1]
def choice(model,tok,task,layer,swap):
 base=base_text(tok,task,swap); idx=len(tok(base,add_special_tokens=False)['input_ids'])-1; return build_fast_cache(model,tok,base+'Choice\n',idx,layer)
def one(model,tok,task,orientation):
 a=tok('A',add_special_tokens=False)['input_ids'][0]; b=tok('B',add_special_tokens=False)['input_ids'][0]; ps=orientation=='semantic_swapped'; co,zo=choice(model,tok,task,14,ps); cs,zs=choice(model,tok,task,14,not ps); z0=torch.cat([zo,zs]); r0,rs0,p0,states,rows=walk(z0,co,cs,a,b); return {'trial_id':task['trial_id'],'orientation':orientation,'baseline':r0,'max_target_error':max(abs(x['original_margin']-x['target']) for x in rows),'max_position_drift':max(abs(x['position_drift']) for x in rows),'negative_margin':rows[0]['original_margin'],'positive_margin':rows[-1]['original_margin'],'pass':max(abs(x['original_margin']-x['target']) for x in rows)<=.150001 and max(abs(x['position_drift']) for x in rows)<=.150001}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--index',type=int,required=True);args=p.parse_args();torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True);model.eval();model.requires_grad_(False);task=load_jsonl(ROOT/'data/validation.jsonl')[args.index]; rows=[one(model,tok,task,o) for o in ['original','semantic_swapped']]; dest=ROOT/'results/validation'/f'choice_{args.index:02d}.json';dest.write_text(json.dumps(rows,indent=2)+'\n');print(dest)
if __name__=='__main__':main()
