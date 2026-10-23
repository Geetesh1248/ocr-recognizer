import cv2
import numpy as np


def clusters(values):
    groups = []
    for value in values:
        if groups and value-groups[-1][-1] <= 3:
            groups[-1].append(int(value))
        else:
            groups.append([int(value)])
    return [round(sum(group)/len(group)) for group in groups]


def extract(image, lines, min_area=1000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    height, width = ink.shape
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, width//30), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, height//30))))
    contours, _ = cv2.findContours(cv2.bitwise_or(horizontal, vertical), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = []
    for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[1]):
        x, y, w, h = cv2.boundingRect(contour)
        if w*h < min_area or w < 30 or h < 30:
            continue
        hroi, vroi = horizontal[y:y+h, x:x+w], vertical[y:y+h, x:x+w]
        xs = [x+i for i in clusters(np.where((vroi > 0).sum(axis=0) >= h*.35)[0])]
        ys = [y+i for i in clusters(np.where((hroi > 0).sum(axis=1) >= w*.35)[0])]
        if len(xs) < 2 or len(ys) < 2:
            continue
        rows, columns = len(ys)-1, len(xs)-1
        if rows*columns > 1000:
            raise ValueError("Detected table grid exceeds 1000 cells")
        parent = list(range(rows*columns))
        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index
        def union(a, b):
            parent[root(a)] = root(b)
        for row in range(rows):
            for col in range(columns):
                index = row*columns+col
                if col+1 < columns:
                    sample = vertical[ys[row]+3:max(ys[row]+4, ys[row+1]-3), max(0, xs[col+1]-2):xs[col+1]+3]
                    present = sample.size and np.any(sample > 0, axis=1).mean() >= .5
                    if not present:
                        union(index, index+1)
                if row+1 < rows:
                    sample = horizontal[max(0, ys[row+1]-2):ys[row+1]+3, xs[col]+3:max(xs[col]+4, xs[col+1]-3)]
                    present = sample.size and np.any(sample > 0, axis=0).mean() >= .5
                    if not present:
                        union(index, index+columns)
        members = {}
        for index in range(rows*columns):
            members.setdefault(root(index), []).append((index//columns, index % columns))
        cells, flags = [], []
        for group in members.values():
            r0, r1 = min(r for r, c in group), max(r for r, c in group)
            c0, c1 = min(c for r, c in group), max(c for r, c in group)
            if len(group) != (r1-r0+1)*(c1-c0+1):
                flags.append("nonrectangular_merge_kept_as_atomic_cells")
                rectangles = [(r, r, c, c) for r, c in group]
            else:
                rectangles = [(r0, r1, c0, c1)]
            for r0, r1, c0, c1 in rectangles:
                cells.append({"row": r0, "column": c0, "row_span": r1-r0+1, "column_span": c1-c0+1,
                              "bbox": [xs[c0], ys[r0], xs[c1+1], ys[r1+1]], "line_ids": [], "text": ""})
        assigned, ambiguous = [], []
        for line in lines:
            a = line["bbox"]
            centre = ((a[0]+a[2])/2, (a[1]+a[3])/2)
            matching = []
            for cell in cells:
                b = cell["bbox"]
                if b[0] <= centre[0] < b[2] and b[1] <= centre[1] < b[3]:
                    area = max(1., (a[2]-a[0])*(a[3]-a[1]))
                    overlap = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
                    if overlap/area >= .8:
                        matching.append(cell)
            if len(matching) == 1:
                cell = matching[0]
                cell["line_ids"].append(line["id"])
                cell["text"] += ("\n" if cell["text"] else "")+line.get("text", "")
                assigned.append(line["id"])
            elif xs[0] <= centre[0] <= xs[-1] and ys[0] <= centre[1] <= ys[-1]:
                ambiguous.append(line["id"])
        tables.append({"id": len(tables), "type": "table", "bbox": [xs[0], ys[0], xs[-1], ys[-1]],
                       "rows": rows, "columns": columns, "cells": sorted(cells, key=lambda c: (c["row"], c["column"])),
                       "assigned_line_ids": assigned, "unassigned_line_ids": ambiguous, "flags": flags})
    return tables


def html(table):
    import html as escape
    output = ["<table>"]
    by_position = {(c["row"], c["column"]): c for c in table["cells"]}
    for row in range(table["rows"]):
        output.append("<tr>")
        for col in range(table["columns"]):
            cell = by_position.get((row, col))
            if cell:
                output.append(f"<td rowspan='{cell['row_span']}' colspan='{cell['column_span']}'>{escape.escape(cell['text'])}</td>")
        output.append("</tr>")
    output.append("</table>")
    return "\n".join(output)
