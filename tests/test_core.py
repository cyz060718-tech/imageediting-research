import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import tempfile
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
from core.instruction_parser import rule_parse
from core import masker, editor, metrics
import dataset

class EditingChecks(unittest.TestCase):
    def test_parser_preserves_shirt_noun(self):
        intent=rule_parse('把上衣换成蓝色')
        self.assertEqual((intent.action,intent.target,intent.new_content),('color','shirt','blue'))
        self.assertIn('full body',rule_parse('把猫换成老虎').edit_prompt)

    def test_parser_removes_context_only(self):
        self.assertEqual(rule_parse('去掉背景里的路灯').target,'street lamp')
        self.assertEqual(rule_parse('去掉图中的猫').target,'cat')

    def test_low_confidence_does_not_expand_into_whole_image(self):
        with self.assertRaises(ValueError):masker.build_masks(np.full((32,32),.01))

    def test_components_split_before_dilation(self):
        p=np.zeros((64,64),dtype=np.float32);p[20:30,10:20]=1;p[20:30,24:34]=1
        masks=masker.build_masks(p,dilate=15)
        self.assertEqual(masks['component_count'],2)
        self.assertEqual(masks['components'][0].sum(),100)

    def test_blend_keeps_pixels_outside_support(self):
        a=Image.new('RGB',(32,32),(30,40,50));b=Image.new('RGB',(32,32),(90,100,110))
        soft=np.zeros((32,32),np.float32);soft[8:24,8:24]=.5
        result=np.asarray(editor.blend(a,b,soft))
        np.testing.assert_array_equal(result[soft==0],np.asarray(a)[soft==0])

    def test_missing_gt_is_not_substituted_by_predicted_mask(self):
        im=Image.new('RGB',(32,32));pred=np.zeros((32,32),np.uint8)
        result=metrics.evaluate(im,im,pred_mask=pred)
        self.assertEqual(result['evaluation_region'],'unavailable')
        self.assertNotIn('background_psnr',result)
        self.assertNotIn('mask_iou',result)

    def test_gt_detects_background_hidden_by_wrong_prediction(self):
        a=np.zeros((32,32,3),np.uint8);b=a.copy();b[20:25,20:25]=100
        gt=np.zeros((32,32),np.uint8);gt[0:8,0:8]=255
        result=metrics.evaluate(Image.fromarray(a),Image.fromarray(b),pred_mask=np.full((32,32),255,np.uint8),gt_mask=gt)
        self.assertGreater(result['background_mae'],0)
        self.assertLess(result['mask_iou'],1)

    def test_empty_background_is_unavailable_not_perfect(self):
        im=Image.new('RGB',(32,32));result=metrics.evaluate(im,im,gt_mask=np.full((32,32),255,np.uint8))
        self.assertIsNone(result['background_psnr'])
        self.assertIsNone(result['background_exact'])

    def test_identical_pixels_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(32,32)).save(root/'a.png')
            (root/'b.png').write_bytes((root/'a.png').read_bytes())
            items=[{'id':'a','split':'dev','image':'a.png','action':'color','instruction':'change jacket to blue','eval_text':'blue jacket'},
                   {'id':'b','split':'test','image':'b.png','action':'color','instruction':'change jacket to blue','eval_text':'blue jacket'}]
            with patch.object(dataset,'DATA',root):report=dataset.validate(items)
            self.assertTrue(any('泄漏' in error for error in report['errors']))

    def test_extreme_aspect_ratio_never_becomes_zero(self):
        self.assertEqual(min(editor.prepare_image(Image.new('RGB',(8,10000))).size),8)

    def test_manual_mask_wrong_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'同尺寸'):
            editor.edit(Image.new('RGB',(32,32)),'把猫换成老虎',manual_mask=Image.new('L',(16,16),255))

    def test_direction_not_silently_lost_when_bundle_is_precomputed(self):
        with self.assertRaisesRegex(ValueError,'方位'):
            editor.edit(Image.new('RGB',(32,32)),'把左边的猫换成老虎',bundle={'source':'clipseg'})

    def test_failed_auto_mask_does_not_prevent_baseline_evaluation(self):
        import eval as experiment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image_path=root/'image.png';Image.new('RGB',(32,32)).save(image_path)
            settings=root/'settings.json';settings.write_text(json.dumps({
                'steps':25,'guidance':7.5,'strength':None,'mask_threshold':.4,'dilate':9,'feather':11,
                'seeds':[42],'methods':['ours','ip2p'],'learned_metrics':False,'warmup':False}))
            item={'id':'sample','image':'image.png','instruction':'把外套换成蓝色','action':'color','split':'dev','eval_text':'blue jacket'}
            im=editor.prepare_image(Image.new('RGB',(32,32)))
            fake={'input':im,'result':im,'raw':im,'mask':None,'time':.1,'peak_vram_mb':None,'strength':1.}
            with patch.object(sys,'argv',['eval.py','--settings',str(settings)]), \
                 patch.object(dataset,'validate',return_value={'errors':[],'warnings':[],'file_hashes':{}}), \
                 patch.object(dataset,'load_items',return_value=[item]), \
                 patch.object(dataset,'resolve',return_value=image_path), \
                 patch.object(editor,'runtime',return_value=(SimpleNamespace(),'cpu')), \
                 patch.object(editor,'get_bundle',side_effect=ValueError('no target')), \
                 patch.object(editor,'edit',return_value=fake), \
                 patch.object(experiment.config,'OUTPUT_DIR',root/'outputs'):
                with self.assertRaises(SystemExit) as stop:experiment.main()
                self.assertEqual(stop.exception.code,1)
            completion=json.loads(next((root/'outputs').rglob('completion.json')).read_text())
            self.assertEqual(completion,{'completed':1,'failed':1})

    def test_paired_comparison_excludes_incomplete_cases_symmetrically(self):
        from eval import paired_rows
        rows=[{'id':'a','seed':42,'method':'ours','status':'ok'},
              {'id':'a','seed':42,'method':'ip2p','status':'ok'},
              {'id':'b','seed':42,'method':'ours','status':'failed'},
              {'id':'b','seed':42,'method':'ip2p','status':'ok'}]
        self.assertEqual([r['id'] for r in paired_rows(rows,['ours','ip2p'])],['a','a'])

if __name__=='__main__':unittest.main()
