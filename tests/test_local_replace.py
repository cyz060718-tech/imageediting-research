"""Geometry and conservative-count regressions for local replacement."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from core import editor,local_replace
from core.grounded_mask import remove_small_islands
from core.instruction_parser import parse
from core.result_check import match_regions


class LocalReplaceChecks(unittest.TestCase):
    def regions(self):
        first=np.zeros((100,160),np.uint8);first[20:70,10:50]=1
        second=np.zeros_like(first);second[30:80,100:140]=1
        return [first,second]

    def test_disconnected_leak_removed_without_dropping_small_separate_instance(self):
        first,second=self.regions()
        first[5:8,70:73]=1
        cleaned=remove_small_islands(first)
        self.assertEqual(int(cleaned[5:8,70:73].sum()),0)
        self.assertEqual(int(cleaned[20:70,10:50].sum()),2000)
        tiny=np.zeros_like(first);tiny[5:9,70:74]=1
        np.testing.assert_array_equal(remove_small_islands(tiny),tiny)

    def test_one_large_detection_cannot_confirm_two_objects(self):
        check=match_regions(self.regions(),[{'box':[5,15,145,85],'score':.9}])
        self.assertEqual(check['confirmed_regions'],1)
        self.assertEqual(check['status'],'needs_review')

    def test_unrelated_new_object_does_not_confirm_replacement(self):
        check=match_regions(self.regions(),[{'box':[60,0,80,15],'score':.9}])
        self.assertEqual(check['confirmed_regions'],0)

    def test_two_matching_detections_confirm_two_regions(self):
        check=match_regions(self.regions(),[{'box':[10,20,50,70]},{'box':[100,30,140,80]}])
        self.assertEqual(check['confirmed_regions'],2)
        self.assertEqual(check['status'],'detected_in_selected_regions')

    def test_crop_replacement_preserves_native_resolution(self):
        source=Image.new('RGB',(803,601))
        self.assertEqual(editor.prepare_for_method(source,parse('把猫换成老虎'),'quality').size,source.size)

    def test_independent_crops_keep_unselected_pixels_exact(self):
        import torch
        from diffusers.image_processor import VaeImageProcessor
        class PaintPipeline:
            image_processor=VaeImageProcessor()
            calls=0
            def __call__(self,**kwargs):
                color='blue' if self.calls==0 else 'green'
                self.calls+=1
                return SimpleNamespace(images=[Image.new('RGB',(512,512),color)])
        pipe=PaintPipeline()
        source=Image.new('RGB',(160,100),(120,80,40))
        result,alpha,_=local_replace.replace(source,{'components':self.regions()},'tigers',pipe,
            torch,'cpu',30,9.,1.,42,3,5)
        self.assertEqual(result.size,source.size)
        self.assertEqual(pipe.calls,2)
        np.testing.assert_array_equal(np.asarray(result)[alpha==0],np.asarray(source)[alpha==0])
        self.assertGreater(np.asarray(result)[40,30,2],200)
        self.assertGreater(np.asarray(result)[50,120,1],100)

    def test_worse_retry_is_rejected_and_does_not_poison_next_retry(self):
        source=Image.new('RGB',(160,100),'white')
        initial=Image.new('RGB',source.size,'blue')
        worse=Image.new('RGB',source.size,'red')
        improved=Image.new('RGB',source.size,'green')
        alpha=np.ones((100,160),np.float32)
        bundle={'components':self.regions(),'source':'mock','hard':np.full((100,160),255,np.uint8),'soft':alpha}
        first={'status':'needs_review','selected_instances':2,'confirmed_regions':1,'unconfirmed_regions':[2]}
        missed={'status':'needs_review','selected_instances':2,'confirmed_regions':0,'unconfirmed_regions':[1,2]}
        passed={'status':'detected_in_selected_regions','selected_instances':2,'confirmed_regions':2,'unconfirmed_regions':[]}
        with patch.object(editor,'runtime',return_value=(SimpleNamespace(),'cpu')), \
                patch.object(editor,'get_pipe',return_value=object()), \
                patch('core.local_replace.replace',side_effect=[
                    (initial,alpha,{'instance_count':2}),(worse,alpha,{'instance_count':1}),
                    (improved,alpha,{'instance_count':1})]) as generate, \
                patch('core.result_check.check_replacement',side_effect=[first,missed,passed]):
            out=editor.edit(source,'把猫换成老虎','quality',bundle=bundle,prepared=True)
        self.assertIs(out['result'],improved)
        self.assertIs(generate.call_args_list[2].args[0],source)
        self.assertIs(generate.call_args_list[2].kwargs['base_result'],initial)
        self.assertEqual(len(generate.call_args_list[1].args[1]['components']),1)
        self.assertFalse(out['engine_info']['attempts'][1]['accepted'])
        self.assertTrue(out['engine_info']['attempts'][2]['accepted'])

    def test_unavailable_detector_does_not_cause_unbounded_generation(self):
        source=Image.new('RGB',(160,100),'white');alpha=np.ones((100,160),np.float32)
        bundle={'components':self.regions(),'source':'mock','hard':np.full((100,160),255,np.uint8),'soft':alpha}
        with patch.object(editor,'runtime',return_value=(SimpleNamespace(),'cpu')), \
                patch.object(editor,'get_pipe',return_value=object()), \
                patch('core.local_replace.replace',return_value=(source,alpha,{'instance_count':2})) as generate, \
                patch('core.result_check.check_replacement',return_value={'status':'unavailable'}):
            editor.edit(source,'把猫换成老虎','quality',bundle=bundle,prepared=True)
        self.assertEqual(generate.call_count,1)

    def test_reference_projection_preserves_native_pixels_outside_its_support(self):
        import torch
        from diffusers.image_processor import VaeImageProcessor
        class PaintPipeline:
            image_processor=VaeImageProcessor()
            def __call__(self,**kwargs):
                return SimpleNamespace(images=[Image.new('RGB',(512,512),'blue')])
        image=Image.new('RGB',(803,601),(120,80,40))
        base=image.copy();base.putpixel((5,5),(1,2,3))
        component=np.zeros((601,803),np.uint8);component[150:450,200:500]=1
        result,alpha,_=local_replace.replace_at_reference_scale(image,{'components':[component]},'tiger',
            PaintPipeline(),torch,'cpu',30,9,1,42,15,21,base_result=base)
        self.assertEqual(result.size,image.size)
        np.testing.assert_array_equal(np.asarray(result)[alpha==0],np.asarray(base)[alpha==0])
        self.assertEqual(result.getpixel((5,5)),(1,2,3))


if __name__=='__main__':unittest.main()
