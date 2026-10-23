"""Build a separate line dataset, preserving existing prepared word data and splits."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from common import ROOT, manifest, read_json, save_json, sha256, image_read, image_write, check_splits
from prepare_funsd import transform_polygon
from increment_02_detection.line_labels import lines_from_entities


def convert_lines(words, raw, output):
    words, raw, output = Path(words).resolve(), Path(raw).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new or empty line dataset output folder')
    source_summary = read_json(words/'preparation_summary.json')
    summary = {'annotation_level': 'line', 'label_source': 'derived_from_FUNSD_words_v1',
               'grouping': 'within entity; vertical overlap; maximum gap 3x text height; vertical-rule separation',
               'limitation': 'Entity boundaries and conservative gaps may split a visual line into fragments. Not native line ground truth.',
               'config': source_summary['config'], 'seed': source_summary['seed'], 'splits': {}}
    for split, raw_split in [('train','training_data'), ('val','training_data'), ('test','testing_data')]:
        records, total, ignored_total, multiword = [], 0, 0, 0
        for record in manifest(words/f'detection_{split}.jsonl'):
            doc = record['document_id']
            annotation = raw/raw_split/'annotations'/f'{doc}.json'
            source = raw/raw_split/'images'/f'{doc}.png'
            details = read_json(words/'transforms'/split/f'{doc}.json')
            if sha256(source) != details['source_sha256'] or sha256(annotation) != details['annotation_sha256']:
                raise ValueError(f'Raw data changed since word preparation: {doc}')
            prepared = image_read(record['image'])
            width, height = details['output_size']
            if [prepared.shape[1],prepared.shape[0]] != [width,height]:
                raise ValueError(f'Prepared dimensions do not match transform: {doc}')
            matrix = details['input_to_prepared_matrix']
            polygons, ignored, line_info = [], [], []
            for line in lines_from_entities(read_json(annotation)['form'], image_read(source)):
                x0,y0,x1,y1 = line['box']
                polygon, visible = transform_polygon([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], matrix, width, height)
                if polygon is None:
                    continue
                flag = bool(line['ignore'] or visible < .9)
                polygons.append(polygon)
                ignored.append(flag)
                line_info.append(dict(line, polygon=polygon, ignore=flag, visible_fraction=visible))
                multiword += int(line['word_count'] > 1 and not flag)
            relative = Path('images')/split/f'{doc}.png'
            (output/relative).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(record['image'], output/relative)
            records.append({'image': relative.as_posix(), 'document_id': doc, 'polygons':polygons,
                            'ignore':ignored, 'annotation_level':'line', 'label_source':summary['label_source']})
            save_json(output/'transforms'/split/f'{doc}.json', details)
            save_json(output/'line_groups'/split/f'{doc}.json', line_info)
            if len(records) <= 3:
                preview = prepared.copy()
                for p, flag in zip(polygons, ignored):
                    cv2.polylines(preview,[np.round(p).astype(np.int32)],True,(0,0,255) if flag else (0,160,0),1)
                image_write(output/'previews'/split/f'{doc}.png',preview)
            total += len(polygons)
            ignored_total += sum(ignored)
        (output/f'detection_{split}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
        summary['splits'][split] = {'pages':len(records),'lines':total,'ignored_lines':ignored_total,'multiword_lines':multiword}
        print(split, summary['splits'][split],flush=True)
    splits = [manifest(output/f'detection_{split}.jsonl') for split in ['train','val','test']]
    for i in range(3):
        for j in range(i+1,3):
            check_splits(splits[i],splits[j])
    save_json(output/'preparation_summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--words',default=str(ROOT/'datasets/funsd_prepared'))
    parser.add_argument('--raw',default=str(ROOT/'datasets/funsd_raw/dataset'))
    parser.add_argument('--output',default=str(ROOT/'datasets/funsd_lines_prepared'))
    args = parser.parse_args()
    convert_lines(args.words,args.raw,args.output)


if __name__ == '__main__':
    main()
