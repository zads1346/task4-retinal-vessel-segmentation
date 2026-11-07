"""Float32 exponential moving average; no labels or validation-dependent updates."""
from copy import deepcopy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.995):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.model = deepcopy(model).float().eval().requires_grad_(False)
        self.decay = float(decay)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        # Avoid a long early lag behind randomly initialized weights.
        decay = min(self.decay, (1+self.num_updates)/(10+self.num_updates))
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            incoming = source[name].detach()
            if torch.is_floating_point(value):
                value.lerp_(incoming.to(value.dtype), 1-decay)
            else:
                value.copy_(incoming)
        self.model.eval()

    def state_dict(self):
        return {'model':self.model.state_dict(), 'decay':self.decay, 'num_updates':self.num_updates,
                'algorithm':'adaptive_decay_v1'}

    def load_state_dict(self, state):
        if state.get('algorithm') != 'adaptive_decay_v1':
            raise ValueError('EMA algorithm changed on resume')
        if float(state['decay']) != self.decay:
            raise ValueError('EMA decay changed on resume')
        self.model.load_state_dict(state['model'])
        self.num_updates = int(state['num_updates'])
        self.model.eval().requires_grad_(False)
