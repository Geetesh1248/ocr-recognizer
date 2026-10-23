import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from common import ROOT, image_write, read_json, manifest
from increment_02_detection.line_labels import lines_from_entities
from prepare_funsd import convert
from prepare_funsd_lines import convert_lines


def word(text, box):
    return {'text':text, 'box':box}


class LineDetectionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((150,300,3),255,np.uint8)

    def test_multiline_entity_becomes_two_lines(self):
        entities = [{'words':[word('Hello',[10,10,40,25]),word('world',[48,11,90,26]),
                              word('Next',[10,40,40,55]),word('line',[48,40,80,55])]}]
        lines = lines_from_entities(entities,self.image)
        self.assertEqual([line['text'] for line in lines],['Hello world','Next line'])
        self.assertEqual(lines[0]['word_count'],2)

    def test_entities_are_not_merged(self):
        entities = [{'words':[word('Cell1',[10,10,40,25])]},
                    {'words':[word('Cell2',[48,10,90,25])]}]
        self.assertEqual(len(lines_from_entities(entities,self.image)),2)

    def test_large_gap_separates_columns_inside_entity(self):
        entities = [{'words':[word('Left',[10,10,40,25]),word('Right',[200,10,240,25])]}]
        self.assertEqual(len(lines_from_entities(entities,self.image)),2)

    def test_visible_vertical_table_border_separates_cells(self):
        cv2.line(self.image,(48,0),(48,80),(0,0,0),2)
        entities = [{'words':[word('Left',[10,20,40,35]),word('Right',[55,20,90,35])]}]
        self.assertEqual(len(lines_from_entities(entities,self.image)),2)

    def test_unknown_word_is_ignored_without_poisoning_known_line(self):
        entities = [{'words':[word('Hello',[10,10,40,25]),word('',[45,10,70,25])]}]
        lines = lines_from_entities(entities,self.image)
        self.assertEqual(sum(line['ignore'] for line in lines),1)
        self.assertEqual(sum(not line['ignore'] for line in lines),1)

    def test_conversion_and_real_line_training(self):
        import torch
        from increment_02_detection.train import train
        from increment_02_detection.inference import Detector
        from evaluate import detection_score
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            raw, words, lines = root/'dataset', root/'words', root/'lines'
            for split,count in [('training_data',3),('testing_data',1)]:
                for i in range(count):
                    stem = f'{split}_{i}'
                    image = np.full((80+i,160,3),250-i-(10 if split=='testing_data' else 0),np.uint8)
                    cv2.putText(image,'HELLO WORLD',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
                    image_write(raw/split/'images'/f'{stem}.png',image)
                    p = raw/split/'annotations'/f'{stem}.json'
                    p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_text(json.dumps({'form':[{'words':[word('HELLO',[10,15,48,32]),word('WORLD',[54,15,98,32])]}]}))
            config = read_json(ROOT/'config.json')
            config['preparation'].update(deskew=False,perspective=False)
            convert(raw,words,config,rotation=90)
            word_bytes = (words/'detection_train.jsonl').read_bytes()
            summary = convert_lines(words,raw,lines)
            self.assertEqual(summary['splits']['train']['multiword_lines'],2)
            self.assertEqual(word_bytes,(words/'detection_train.jsonl').read_bytes())
            for split in ['train','val','test']:
                self.assertEqual([r['document_id'] for r in manifest(words/f'detection_{split}.jsonl')],
                                 [r['document_id'] for r in manifest(lines/f'detection_{split}.jsonl')])
            checkpoint = root/'line.pt'
            train(lines/'detection_train.jsonl',lines/'detection_val.jsonl',checkpoint,epochs=1,size=64)
            settings = dict(config['detector'],annotation_level='line')
            self.assertEqual(Detector(checkpoint,settings).annotation_level,'line')
            with self.assertRaises(ValueError):
                Detector(checkpoint,dict(settings,annotation_level='word'))
            with self.assertRaises(ValueError):
                detection_score(checkpoint,words/'detection_test.jsonl',settings)
            with self.assertRaises(ValueError):
                train(words/'detection_train.jsonl',lines/'detection_val.jsonl',root/'bad.pt',epochs=1,size=64)
