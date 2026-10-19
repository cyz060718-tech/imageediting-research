import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from core import removal


class RemovalChecks(unittest.TestCase):
    def test_odd_native_size_and_unselected_pixels_are_preserved(self):
        image=Image.new('RGB',(101,83),(180,80,30))
        mask=np.zeros((83,101),np.uint8);mask[15:65,20:70]=255
        alpha=(mask>0).astype(np.float32)
        def repair(source,selected):
            result=torch.zeros_like(source);result[:,2]=1
            return result
        with patch.object(removal,'load',return_value=repair):
            edited,_=removal.remove(image,mask,alpha)
        self.assertEqual(edited.size,image.size)
        np.testing.assert_array_equal(np.asarray(edited)[alpha==0],np.asarray(image)[alpha==0])
        self.assertEqual(tuple(np.asarray(edited)[30,40]),(0,0,255))

    def test_empty_selection_rejected_before_loading_model(self):
        image=Image.new('RGB',(64,64));empty=np.zeros((64,64),np.uint8)
        with patch.object(removal,'load') as load:
            with self.assertRaisesRegex(ValueError,'删除选区'):
                removal.remove(image,empty,empty.astype(np.float32))
        load.assert_not_called()


if __name__=='__main__':unittest.main()
