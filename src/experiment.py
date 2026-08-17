import hashlib,json,torch
from .fast_suffix import build_fast_cache
TARGETS=torch.tensor([-3.0,-1.5,-0.5,-0.1,0.1,0.5,1.5,3.0],dtype=torch.float32)
SYSTEM='A cue word will appear at the start of your response. After Choice, output exactly A or B. After ConfidenceNumber, output exactly one digit from 1 to 5, where 1 means almost indifferent and 5 means extremely confident. After ConfidenceWord, output exactly low, medium, or high, where low means almost indifferent and high means extremely confident. Output nothing else.'
def base_text(tokenizer,task,swap=False):
 a,b=(task['option_b'],task['option_a']) if swap else (task['option_a'],task['option_b'])
 u=f"Option A is {a}. Option B is {b}. Consider which option you currently prefer."
 return tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':u}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
def build(model,tokenizer,task,layer,swap=False):
 base=base_text(tokenizer,task,swap); idx=len(tokenizer(base,add_special_tokens=False)['input_ids'])-1
 c,zc=build_fast_cache(model,tokenizer,base+'Choice\n',idx,layer)
 n,zn=build_fast_cache(model,tokenizer,base+'ConfidenceNumber\n',idx,layer)
 w,zw=build_fast_cache(model,tokenizer,base+'ConfidenceWord\n',idx,layer)
 return c,n,w,zc,zn,zw,max(float((zc-zn).abs().max()),float((zc-zw).abs().max()))

def build_choice(model,tokenizer,task,layer,swap=False):
 base=base_text(tokenizer,task,swap); idx=len(tokenizer(base,add_special_tokens=False)['input_ids'])-1
 return build_fast_cache(model,tokenizer,base+'Choice\n',idx,layer)

def pair_batch(zs,co,cs,a,b):
 d=zs.shape[1]//2; lo=co.logits_batch(zs[:,:d]); ls=cs.logits_batch(zs[:,d:]); ro=lo[:,b]-lo[:,a]; rs=ls[:,b]-ls[:,a]; p=.5*(ro+rs); c=.5*(ro-rs); return ro,rs,p,c,lo,ls
def project(v,rows,ridge=1e-10):
 m=torch.stack(rows); eye=torch.eye(m.shape[0],dtype=m.dtype,device=m.device); return v-m.T@torch.linalg.solve(m@m.T+ridge*eye,m@v)
def walk(z0,co,cs,a,b,targets=TARGETS,tol=.15,max_iter=12):
 with torch.no_grad(): r0,rs0,p0,_,_,_=pair_batch(z0[None],co,cs,a,b)
 zs=z0[None].repeat(len(targets),1).detach(); t=targets.to(zs); k=-1
 for k in range(max_iter):
  z=zs.detach().requires_grad_(True); ro,_,p,_,_,_=pair_batch(z,co,cs,a,b); er=t-ro; ep=p0[0]-p; done=(er.abs()<=tol)&(ep.abs()<=tol)
  if bool(done.all()): break
  gr=torch.autograd.grad(ro.sum(),z,retain_graph=True)[0]; gp=torch.autograd.grad(p.sum(),z)[0]
  a11=(gr*gr).sum(1)+1e-8; a12=(gr*gp).sum(1); a22=(gp*gp).sum(1)+1e-8; det=a11*a22-a12*a12
  w1=er.clamp(-1.5,1.5); w2=ep.clamp(-.75,.75); alpha=(a22*w1-a12*w2)/det; beta=(-a12*w1+a11*w2)/det
  delta=alpha[:,None]*gr+beta[:,None]*gp; cap=.08*torch.linalg.vector_norm(z,dim=1); dn=torch.linalg.vector_norm(delta,dim=1); delta=delta*torch.minimum(torch.ones_like(dn),cap/(dn+1e-12))[:,None]; delta[done]=0; zs=(z+delta).detach()
 with torch.no_grad(): ro,rs,p,c,lo,ls=pair_batch(zs,co,cs,a,b)
 rows=[]
 for i,target in enumerate(t.tolist()):
  pa=torch.softmax(lo[i][torch.tensor([a,b])],0); rows.append({'target':target,'iterations':k+1,'original_margin':float(ro[i]),'swapped_margin':float(rs[i]),'position':float(p[i]),'position_drift':float(p[i]-p0[0]),'semantic':float(c[i]),'forced_choice':'B' if float(pa[1])>float(pa[0]) else 'A','forced_choice_B_probability':float(pa[1]),'delta_norm':float((zs[i]-z0).norm())})
 return float(r0[0]),float(rs0[0]),float(p0[0]),zs,rows
def add_confidence(rows,zs,num_cache,word_cache,digit_ids,word_ids):
 d=zs.shape[1]//2
 with torch.no_grad(): ln=num_cache.logits_batch(zs[:,:d]); lw=word_cache.logits_batch(zs[:,:d])
 valsn=torch.arange(1,6,dtype=torch.float32); valsw=torch.tensor([1.,3.,5.])
 for i in range(len(rows)):
  pn=torch.softmax(ln[i][digit_ids],0); pw=torch.softmax(lw[i][word_ids],0); fn=torch.softmax(ln[i],0); fw=torch.softmax(lw[i],0)
  rows[i].update({'confidence_number':float(torch.dot(valsn,pn)),'confidence_number_mass':float(fn[digit_ids].sum()),'confidence_word':float(torch.dot(valsw,pw)),'confidence_word_mass':float(fw[word_ids].sum())})
def sham(z0,zs,co,cs,num_cache,word_cache,a,b,digit_ids,word_ids,seed_text,p0):
 z=z0.detach().requires_grad_(True); _,_,p,c,_,_=pair_batch(z[None],co,cs,a,b); gc=torch.autograd.grad(c.sum(),z,retain_graph=True)[0]; gp=torch.autograd.grad(p.sum(),z)[0]
 gen=torch.Generator().manual_seed(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16],16)%(2**31-1)); r=torch.randn(z0.shape,generator=gen,dtype=z0.dtype); r=project(r,[gc,gp]); r=(r/(r.norm()+1e-12)).detach(); norms=torch.linalg.vector_norm(zs-z0[None],dim=1); sh=z0[None]+norms[:,None]*r[None]; d=z0.numel()//2
 with torch.no_grad(): ro,rs,p,c,lo,_=pair_batch(sh,co,cs,a,b); ln=num_cache.logits_batch(sh[:,:d]); lw=word_cache.logits_batch(sh[:,:d])
 valsn=torch.arange(1,6,dtype=torch.float32); valsw=torch.tensor([1.,3.,5.]); out=[]
 for i in range(len(zs)):
  pn=torch.softmax(ln[i][digit_ids],0); pw=torch.softmax(lw[i][word_ids],0); fn=torch.softmax(ln[i],0); fw=torch.softmax(lw[i],0); pc=torch.softmax(lo[i][torch.tensor([a,b])],0)
  out.append({'matched_target':float(TARGETS[i]),'original_margin':float(ro[i]),'swapped_margin':float(rs[i]),'position':float(p[i]),'position_drift':float(p[i]-p0),'forced_choice':'B' if float(pc[1])>float(pc[0]) else 'A','confidence_number':float(torch.dot(valsn,pn)),'confidence_number_mass':float(fn[digit_ids].sum()),'confidence_word':float(torch.dot(valsw,pw)),'confidence_word_mass':float(fw[word_ids].sum()),'delta_norm':float(norms[i])})
 return out
def run_orientation(model,tokenizer,task,layer,orientation):
 a=tokenizer('A',add_special_tokens=False)['input_ids'][0]; b=tokenizer('B',add_special_tokens=False)['input_ids'][0]; dids=[tokenizer(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,6)]; wids=[tokenizer(x,add_special_tokens=False)['input_ids'][0] for x in ['low','medium','high']]
 primary_swap=orientation=='semantic_swapped'; co,cn,cw,zo,zn,zw,e0=build(model,tokenizer,task,layer,primary_swap); cs,zs=build_choice(model,tokenizer,task,layer,not primary_swap); z0=torch.cat([zo,zs]); r0,rs0,p0,states,rows=walk(z0,co,cs,a,b); add_confidence(rows,states,cn,cw,dids,wids); sh=sham(z0,states,co,cs,cn,cw,a,b,dids,wids,task['trial_id']+'_'+orientation,p0)
 return {'trial_id':task['trial_id'],'family':task['family'],'orientation':orientation,'layer':layer,'common_prefix_max_abs_error':e0,'baseline_original_margin':r0,'baseline_counterfactual_margin':rs0,'baseline_position':p0,'active':rows,'sham':sh}
def load_jsonl(path): return [json.loads(x) for x in open(path) if x.strip()]
