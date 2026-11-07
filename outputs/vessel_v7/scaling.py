"""Parameter-space shrinkage of v6's saved LoRA update; no probability mixing."""
import torch

TARGETS={f'decoders.{block}.layers.{layer}.weight' for block in range(4) for layer in (0,3)}


def scaled_state(base,adapted,beta):
    if beta not in (0.,.25,.5,.75,1.):raise ValueError('Unexpected scaling grid')
    if base.keys()!=adapted.keys():raise ValueError('Model state mismatch')
    result={}
    for name,value in base.items():
        incoming=adapted[name]
        if value.shape!=incoming.shape or not torch.isfinite(incoming).all():raise ValueError(name)
        if name not in TARGETS and not torch.equal(value,incoming):raise ValueError('Non-LoRA parameter changed: '+name)
        if beta==0:result[name]=value.clone()
        elif beta==1:result[name]=incoming.clone()
        else:result[name]=value+float(beta)*(incoming-value)
    return result
