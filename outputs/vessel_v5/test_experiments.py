import unittest
import numpy as np
import torch
from experiments import transform_cases, soft_cldice, experiment_loss
from thin_loss import thin_aware_loss


class TreatmentTests(unittest.TestCase):
    def test_preprocess_preserves_labels_raw_and_red_blue(self):
        rng=np.random.default_rng(7)
        rgb=rng.random((64,64,3),dtype=np.float32)
        rgb[:5]=0
        mask=np.zeros((64,64),np.float32)
        cases={'1':{'image':rgb,'mask':mask}}
        saved=rgb.copy()
        a=transform_cases(cases,'A')
        self.assertTrue(np.array_equal(rgb,saved))
        self.assertIs(a['1']['mask'],mask)
        self.assertTrue(np.array_equal(a['1']['image'][...,[0,2]],rgb[...,[0,2]]))
        self.assertTrue(np.array_equal(a['1']['image'][:5],rgb[:5]))
        self.assertFalse(np.array_equal(a['1']['image'][...,1],rgb[...,1]))
        self.assertTrue(np.array_equal(a['1']['image'],transform_cases(cases,'A')['1']['image']))
        self.assertIs(transform_cases(cases,'B'),cases)

    def test_cldice_rewards_connected_correct_line(self):
        target=torch.zeros(1,1,32,32)
        target[:,:,16,4:28]=1
        good=soft_cldice(target,target)
        broken=target.clone();broken[:,:,16,13:19]=0
        false=target.clone();false[:,:,4,4:28]=1
        self.assertLess(abs(good.item()),1e-5)
        self.assertGreater(soft_cldice(broken,target).item(),good.item()+0.01)
        self.assertGreater(soft_cldice(false,target).item(),good.item()+0.01)

    def test_empty_and_full_are_finite(self):
        for value in (0.,1.):
            target=torch.full((2,1,32,32),value)
            logits=torch.zeros_like(target,requires_grad=True)
            loss=experiment_loss(logits,target,target,target,'B')
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_A_loss_exact_baseline_and_B_gpu_gradient(self):
        torch.manual_seed(8)
        target=(torch.rand(2,1,32,32)>0.8).float()
        logits=torch.randn_like(target)
        self.assertEqual(experiment_loss(logits,target,target,target,'A').item(),
                         thin_aware_loss(logits,target,target,target).item())
        logits=logits.cuda().requires_grad_();target=target.cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            loss=experiment_loss(logits,target,target,target,'B')
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(),0)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
