import unittest
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from inference import write_json
from model import UNet
from scaling import scaled_state,TARGETS


class ScalingTests(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        base=UNet(4).state_dict();changed={k:v.clone() for k,v in base.items()}
        for k in TARGETS:changed[k]+=0.01
        for beta in (0.,.25,.5,.75,1.):
            result=scaled_state(base,changed,beta)
            for k in base:
                if beta==0:torch.testing.assert_close(result[k],base[k],rtol=0,atol=0)
                elif beta==1:torch.testing.assert_close(result[k],changed[k],rtol=0,atol=0)
                elif k in TARGETS:torch.testing.assert_close(result[k],base[k]+beta*(changed[k]-base[k]),rtol=0,atol=0)
                else:torch.testing.assert_close(result[k],base[k],rtol=0,atol=0)

    def test_rejects_non_decoder_change(self):
        base=UNet(4).state_dict();changed={k:v.clone() for k,v in base.items()}
        changed['output.bias']+=1
        with self.assertRaises(ValueError):scaled_state(base,changed,.5)


class StatusWriteTests(unittest.TestCase):
    def test_transient_windows_reader_lock(self):
        original=Path.replace
        attempts=[]
        def temporarily_locked(source,target):
            attempts.append(1)
            if len(attempts)<3:raise PermissionError('simulated Windows reader lock')
            return original(source,target)
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/'live.json'
            destination.write_text('{"batch":1}',encoding='utf-8')
            with patch.object(Path,'replace',temporarily_locked):
                write_json(destination,{'batch':2})
            self.assertEqual(len(attempts),3)
            self.assertEqual(json.loads(destination.read_text(encoding='utf-8')),{'batch':2})
            self.assertFalse((Path(folder)/'live.json.tmp').exists())


if __name__=='__main__':unittest.main()
