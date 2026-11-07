"""Conv2d LoRA: flatten W[out,in,k,k], train BA, freeze the v4 backbone."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRAConv2d(nn.Module):
    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        if base.groups!=1 or base.padding_mode!='zeros' or rank<1:
            raise ValueError('Only dense, zero-padded convolutions are supported')
        self.base=base.requires_grad_(False)
        self.rank=rank
        self.scale=float(alpha)/rank
        self.A=nn.Parameter(base.weight.new_empty(rank,base.weight[0].numel()))
        self.B=nn.Parameter(base.weight.new_zeros(base.out_channels,rank))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))

    def delta(self):
        # Keep the low-rank product in float32; only convolution uses AMP.
        with torch.autocast(self.A.device.type,enabled=False):
            return (self.B.float()@self.A.float()).reshape_as(self.base.weight)*self.scale

    def forward(self,x):
        return F.conv2d(x,self.base.weight+self.delta(),self.base.bias,
                        self.base.stride,self.base.padding,self.base.dilation,self.base.groups)


def inject(model,rank=4,alpha=4):
    model.requires_grad_(False)
    names=[]
    for name,module in list(model.named_modules()):
        if name.startswith('decoders.') and isinstance(module,nn.Conv2d):
            parent,child=name.rsplit('.',1)
            model.get_submodule(parent)._modules[child]=LoRAConv2d(module,rank,alpha)
            names.append(name)
    if len(names)!=8:raise ValueError(f'Expected 8 decoder convolutions, got {names}')
    return names


def adapter_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}


def load_adapter(model,state):
    expected=set(adapter_state(model))
    if set(state)!=expected:raise ValueError('Adapter keys mismatch')
    params=dict(model.named_parameters())
    with torch.no_grad():
        for k,v in state.items():
            if v.shape!=params[k].shape:raise ValueError('Adapter shape mismatch')
            params[k].copy_(v)


def merged(model):
    result=copy.deepcopy(model).eval()
    for name,module in list(result.named_modules()):
        if isinstance(module,LoRAConv2d):
            conv=copy.deepcopy(module.base)
            with torch.no_grad():conv.weight.add_(module.delta())
            parent,child=name.rsplit('.',1)
            result.get_submodule(parent)._modules[child]=conv
    return result


def frozen_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad}
