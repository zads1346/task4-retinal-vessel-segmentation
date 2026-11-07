import unittest
import torch
from model import UNet
from lora import inject,adapter_state,load_adapter,merged,frozen_state


class LoRATests(unittest.TestCase):
    def test_zero_identity_update_freezing_merge_and_reload(self):
        torch.manual_seed(42)
        model=UNet(4).eval()
        x=torch.rand(2,3,32,32)
        reference=model(x).detach()
        base_state={k:v.clone() for k,v in model.state_dict().items()}
        self.assertEqual(len(inject(model)),8)
        torch.testing.assert_close(model(x),reference,rtol=0,atol=0)
        frozen=frozen_state(model)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            loss=(model(x)-torch.ones_like(reference)).square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
            optimizer.step()
        self.assertGreater(sum(v.abs().sum().item() for k,v in adapter_state(model).items() if k.endswith('.B')),0)
        for k,v in frozen_state(model).items():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)
        torch.testing.assert_close(merged(model)(x),model(x),rtol=1e-5,atol=1e-6)
        restored=UNet(4).eval();restored.load_state_dict(base_state);inject(restored)
        load_adapter(restored,adapter_state(model))
        torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
        with self.assertRaises(ValueError):load_adapter(restored,{})

    def test_cuda_amp(self):
        model=UNet(4).cuda();inject(model)
        with torch.autocast('cuda',dtype=torch.float16):loss=model(torch.rand(2,3,32,32,device='cuda')).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
