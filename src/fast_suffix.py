import torch
from dataclasses import dataclass
from transformers.models.qwen3.modeling_qwen3 import rotate_half, repeat_kv

@dataclass
class FastSuffixCache:
    suffix0: torch.Tensor
    index: int
    layer: int
    model: object
    prefix_kv: list
    cos: torch.Tensor
    sin: torch.Tensor
    baseline_logits: torch.Tensor

    def logits(self, z):
        return self.logits_batch(z[None])[0]

    def logits_batch(self, zs):
        b = zs.shape[0]
        x = self.suffix0.expand(b, -1, -1).clone()
        x[:, 0] = zs.to(x.dtype)
        slen = x.shape[1]
        for off, i in enumerate(range(self.layer + 1, self.model.config.num_hidden_layers)):
            layer = self.model.model.layers[i]
            residual = x
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            ishape = h.shape[:-1]
            hshape = (*ishape, -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(h).view(hshape)).transpose(1,2)
            ks = attn.k_norm(attn.k_proj(h).view(hshape)).transpose(1,2)
            vs = attn.v_proj(h).view(hshape).transpose(1,2)
            c = self.cos[:, self.index:self.index+slen].unsqueeze(1)
            s = self.sin[:, self.index:self.index+slen].unsqueeze(1)
            q = q*c + rotate_half(q)*s
            ks = ks*c + rotate_half(ks)*s
            kp, vp = self.prefix_kv[off]
            kp = kp.expand(b, -1, -1, -1)
            vp = vp.expand(b, -1, -1, -1)
            k = torch.cat([kp, ks], dim=2)
            v = torch.cat([vp, vs], dim=2)
            k = repeat_kv(k, attn.num_key_value_groups)
            v = repeat_kv(v, attn.num_key_value_groups)
            scores = torch.matmul(q, k.transpose(2,3)) * attn.scaling
            total = self.index + slen
            allowed = torch.arange(total, device=x.device)[None,:] <= (self.index + torch.arange(slen, device=x.device)[:,None])
            mask = torch.where(allowed, torch.zeros((),device=x.device,dtype=scores.dtype), torch.full((),float('-inf'),device=x.device,dtype=scores.dtype))
            scores = scores + mask[None,None]
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            ao = torch.matmul(probs, v).transpose(1,2).contiguous().reshape(*ishape, -1)
            ao = attn.o_proj(ao)
            x = residual + ao
            residual = x
            x = residual + layer.mlp(layer.post_attention_layernorm(x))
        x = self.model.model.norm(x)
        return self.model.lm_head(x)[:, -1].float()


def build_fast_cache(model, tokenizer, text, common_index, layer):
    enc = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    full0 = out.hidden_states[layer+1].detach()
    suffix0 = full0[:, common_index:].clone()
    seq = full0.shape[1]
    pos = torch.arange(seq, device=full0.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(full0, pos)
    prefix_kv=[]
    with torch.no_grad():
        for i in range(layer+1, model.config.num_hidden_layers):
            lyr=model.model.layers[i]; attn=lyr.self_attn
            hp=out.hidden_states[i][:,:common_index].detach()
            hp=lyr.input_layernorm(hp)
            shape=(*hp.shape[:-1],-1,attn.head_dim)
            kp=attn.k_norm(attn.k_proj(hp).view(shape)).transpose(1,2)
            vp=attn.v_proj(hp).view(shape).transpose(1,2)
            c=cos[:,:common_index].unsqueeze(1);s=sin[:,:common_index].unsqueeze(1)
            kp=kp*c+rotate_half(kp)*s
            prefix_kv.append((kp.detach(),vp.detach()))
    return FastSuffixCache(suffix0,common_index,layer,model,prefix_kv,cos.detach(),sin.detach(),out.logits[0,-1].float().detach()), full0[0,common_index].float().detach().clone()
