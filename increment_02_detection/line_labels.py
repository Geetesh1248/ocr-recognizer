"""Derive conservative text lines from FUNSD word boxes, without an OCR model.

Lines stay inside each annotated entity. Geometry splits multiline entities and
large gaps; visible vertical rules keep neighbouring table cells separate.
These are derived labels, not human-annotated line ground truth.
"""
import cv2
import numpy as np


def lines_from_entities(entities, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    result = []
    for entity_index, entity in enumerate(entities):
        known, unknown = [], []
        for word in entity['words']:
            box = np.asarray(word['box'], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError('FUNSD word boxes require four finite coordinates')
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                continue
            entry = {'box': box.tolist(), 'text': word.get('text', '').strip()}
            (known if entry['text'] else unknown).append(entry)
        typical = float(np.median([w['box'][3]-w['box'][1] for w in known])) if known else 10.
        vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, round(typical*2)))))
        rows = []
        for word in sorted(known, key=lambda w: ((w['box'][1]+w['box'][3])/2, w['box'][0])):
            x0, y0, x1, y1 = word['box']
            candidates = []
            for index, row in enumerate(rows):
                top, bottom = np.median([[w['box'][1], w['box'][3]] for w in row], axis=0)
                overlap = max(0., min(y1, bottom)-max(y0, top))
                small, large = min(y1-y0, bottom-top), max(y1-y0, bottom-top)
                distance = abs((y0+y1-top-bottom)/2)
                if large <= 2.5*small and overlap >= .5*small and distance <= .5*large:
                    candidates.append((distance, index))
            if candidates:
                rows[min(candidates)[1]].append(word)
            else:
                rows.append([word])
        groups = []
        for row in rows:
            ordered = sorted(row, key=lambda w: w['box'][0])
            group = [ordered[0]]
            for word in ordered[1:]:
                previous = group[-1]['box']
                current = word['box']
                gap = current[0]-previous[2]
                height = float(np.median([w['box'][3]-w['box'][1] for w in group]+[current[3]-current[1]]))
                left, right = max(0, int(previous[2])+1), min(image.shape[1], int(current[0]))
                top, bottom = max(0, int(min(previous[1],current[1]))), min(image.shape[0], int(max(previous[3],current[3])))
                strip = vertical[top:bottom, left:right]
                barrier = strip.size > 0 and np.max(np.count_nonzero(strip, axis=0)) >= .6*max(1,bottom-top)
                if gap > 3*height or barrier:
                    groups.append(group)
                    group = []
                group.append(word)
            groups.append(group)
        for group in groups+[[word] for word in unknown]:
            boxes = np.asarray([w['box'] for w in group])
            result.append({'box': [float(boxes[:,0].min()), float(boxes[:,1].min()),
                                   float(boxes[:,2].max()), float(boxes[:,3].max())],
                           'text': ' '.join(w['text'] for w in group),
                           'word_count': len(group), 'entity_index': entity_index,
                           'ignore': not all(w['text'] for w in group)})
    return sorted(result, key=lambda line: (line['box'][1], line['box'][0]))
