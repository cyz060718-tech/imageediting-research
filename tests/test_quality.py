import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import cv2
import numpy as np
from PIL import Image
from core import appearance,editor

class AppearanceChecks(unittest.TestCase):
    def fixture(self):
        source=np.full((80,96,3),180,np.uint8)
        source[12:70,16:80]=(210,30,30)
        source[15:67,40:46]=(245,245,245)
        source[15:67,70:75]=(20,20,20)
        mask=np.zeros((80,96),np.uint8);mask[12:70,16:80]=255
        return Image.fromarray(source),mask

    def test_color_preserves_protected_pixels(self):
        image,mask=self.fixture();result,alpha,_=appearance.recolor(image,mask,'blue')
        np.testing.assert_array_equal(np.asarray(result)[alpha==0],np.asarray(image)[alpha==0])

    def test_white_stripes_and_black_trim_are_preserved(self):
        image,mask=self.fixture();result,_,info=appearance.recolor(image,mask,'blue')
        self.assertTrue(info['source_hue_gate'])
        np.testing.assert_array_equal(np.asarray(result)[20:60,40:46],np.asarray(image)[20:60,40:46])
        np.testing.assert_array_equal(np.asarray(result)[20:60,70:75],np.asarray(image)[20:60,70:75])

    def test_red_region_really_becomes_blue(self):
        image,mask=self.fixture();result,_,_=appearance.recolor(image,mask,'blue')
        hsv=cv2.cvtColor(np.asarray(result),cv2.COLOR_RGB2HSV)
        patch=hsv[20:60,20:35]
        self.assertTrue(np.all((patch[...,0]>95)&(patch[...,0]<135)&(patch[...,1]>100)))

    def test_midtone_texture_lightness_is_retained(self):
        image,mask=self.fixture();result,_,_=appearance.recolor(image,mask,'blue')
        diagnostics=appearance.diagnostics(image,result,mask,'blue')
        self.assertLess(diagnostics['lightness_mae'],2.)
        self.assertGreater(diagnostics['lightness_correlation'],.99)

    def test_white_object_can_acquire_visible_color(self):
        image=Image.new('RGB',(32,32),'white');mask=np.full((32,32),255,np.uint8)
        result,_,_=appearance.recolor(image,mask,'blue')
        pixel=cv2.cvtColor(np.asarray(result),cv2.COLOR_RGB2HSV)[16,16]
        self.assertTrue(95<int(pixel[0])<135)
        self.assertGreater(int(pixel[1]),40)

    def test_quality_color_retains_native_size_and_is_deterministic(self):
        image=Image.new('RGB',(803,601),(200,30,30));mask=Image.new('L',image.size,0)
        mask.paste(255,(100,100,700,500))
        first=editor.edit(image,'把外套换成蓝色',method='quality',manual_mask=mask,seed=42)
        second=editor.edit(image,'把外套换成蓝色',method='quality',manual_mask=mask,seed=123)
        self.assertEqual(first['result'].size,image.size)
        self.assertEqual(first['effective_steps'],0)
        self.assertEqual(first['engine'],'lab_chroma')
        np.testing.assert_array_equal(np.asarray(first['result']),np.asarray(second['result']))

    def test_unknown_color_is_explicit_error(self):
        image,mask=self.fixture()
        with self.assertRaises(ValueError):appearance.recolor(image,mask,'mystery color')

    def test_painted_region_uses_alpha_and_rejects_stale_image(self):
        import app
        image=Image.new('RGB',(32,32),(30,40,50))
        canvas=app.load_paint(image)
        layer=Image.new('RGBA',(32,32),(0,0,0,0));layer.paste((255,60,60,255),(8,8,24,24))
        canvas['layers']=[layer]
        mask=app.use_paint(image,canvas)
        self.assertEqual(int((np.asarray(mask)>127).sum()),256)
        with self.assertRaises(Exception):app.use_paint(Image.new('RGB',(32,32),'white'),canvas)

    def test_nested_duplicate_masks_do_not_count_as_new_instances(self):
        from core.grounded_mask import deduplicate_instances
        a=np.zeros((48,48),np.uint8);a[2:24,2:24]=1
        duplicate=np.zeros_like(a);duplicate[4:16,4:16]=1
        b=np.zeros_like(a);b[28:44,28:44]=1
        components,details=deduplicate_instances([a,duplicate,b],[{'id':0},{'id':1},{'id':2}])
        self.assertEqual([x['id'] for x in details],[0,2])

    def test_mask_cache_does_not_reuse_a_different_image(self):
        from core import grounded_mask
        from unittest.mock import patch
        grounded_mask._predictions.clear()
        bundle={'source':'mock'}
        with patch.object(grounded_mask,'_generate',return_value=bundle) as prediction:
            first=grounded_mask.generate(Image.new('RGB',(24,24),'red'),'cat')
            again=grounded_mask.generate(Image.new('RGB',(24,24),'red'),'cat')
            different=grounded_mask.generate(Image.new('RGB',(24,24),'blue'),'cat')
            self.assertFalse(first['cache_hit']);self.assertTrue(again['cache_hit']);self.assertFalse(different['cache_hit'])
            self.assertEqual(prediction.call_count,2)
        grounded_mask._predictions.clear()

if __name__=='__main__':unittest.main()
