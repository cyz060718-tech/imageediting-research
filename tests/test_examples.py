"""Chinese UI examples must retain the intended target and editing action."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import app
import dataset
from core.instruction_parser import rule_parse


class ChineseExampleChecks(unittest.TestCase):
    def test_every_dev_example_has_chinese_text_and_matching_edit_intent(self):
        items = [item for item in dataset.load_items() if item['split'] == 'dev']
        self.assertEqual(len(app.EXAMPLES), len(items))
        aliases = {'sofa': 'couch', 'chairs': 'chair', 'armchairs': 'armchair'}
        def canonical(word):
            return aliases.get(word, word)
        for item, (image, instruction) in zip(items, app.EXAMPLES):
            with self.subTest(sample=item['id']):
                self.assertEqual(image, str(dataset.resolve(item['image'])))
                self.assertEqual(instruction, item['example_instruction_zh'])
                self.assertRegex(instruction, r'[\u4e00-\u9fff]')
                self.assertNotRegex(instruction, r'[A-Za-z]')
                original = rule_parse(item['instruction'])
                chinese = rule_parse(instruction)
                self.assertTrue(chinese.target)
                self.assertEqual(chinese.action, original.action)
                self.assertEqual(canonical(chinese.target), canonical(original.target))
                self.assertEqual(canonical(chinese.new_content), canonical(original.new_content))


if __name__ == '__main__':
    unittest.main()
