import statistics


def gaps(items, axis):
    intervals = sorted((line["bbox"][axis], line["bbox"][axis+2]) for line in items)
    end, output = intervals[0][1], []
    for start, finish in intervals[1:]:
        if start > end:
            output.append((start-end, (start+end)/2))
        end = max(end, finish)
    return output


def reading_order(lines, column_gap=30, depth=0):
    """Recursive whitespace cuts: horizontal sections, then side-by-side columns."""
    if len(lines) <= 1 or depth > 30:
        return list(lines)
    heights = [max(1., b["bbox"][3]-b["bbox"][1]) for b in lines]
    candidates = [(gap, point, 1) for gap, point in gaps(lines, 1) if gap > statistics.median(heights)*1.5]
    if not candidates:
        candidates = [(gap, point, 0) for gap, point in gaps(lines, 0) if gap >= column_gap]
    if candidates:
        _, cut, axis = max(candidates)
        left = [line for line in lines if line["bbox"][axis+2] <= cut]
        right = [line for line in lines if line["bbox"][axis] >= cut]
        if left and right and len(left)+len(right) == len(lines):
            return reading_order(left, column_gap, depth+1)+reading_order(right, column_gap, depth+1)
    return sorted(lines, key=lambda b: (b["bbox"][1], b["bbox"][0], b["id"]))


def blocks(lines, column_gap=30):
    if not lines:
        return []
    ordered = reading_order(lines, column_gap)
    median_height = statistics.median(max(1., b["bbox"][3]-b["bbox"][1]) for b in lines)
    groups = []
    for line in ordered:
        box = line["bbox"]
        heading = box[3]-box[1] > median_height*1.5 and len(line.get("text", "").split()) <= 12
        join = False
        if groups and not heading and groups[-1]["type"] == "paragraph":
            previous = groups[-1]["lines"][-1]["bbox"]
            overlap = max(0., min(previous[2], box[2])-max(previous[0], box[0]))
            denominator = max(1., min(previous[2]-previous[0], box[2]-box[0]))
            vertical_gap = box[1]-previous[3]
            join = -.5*median_height <= vertical_gap <= median_height*1.5 and overlap/denominator >= .25
        if join:
            groups[-1]["lines"].append(line)
        else:
            groups.append({"type": "heading" if heading else "paragraph", "lines": [line]})
    output = []
    for index, group in enumerate(groups):
        members = group["lines"]
        output.append({"id": index, "type": group["type"], "line_ids": [m["id"] for m in members],
                       "bbox": [min(m["bbox"][0] for m in members), min(m["bbox"][1] for m in members),
                                max(m["bbox"][2] for m in members), max(m["bbox"][3] for m in members)],
                       "text": "\n".join(m.get("text", "") for m in members)})
    return output
