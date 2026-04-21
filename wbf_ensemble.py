import json
from ensemble_boxes import weighted_boxes_fusion
import os

# ─────────────────────────── Config ────────────────────────────
FILES = [
    "pred_log1.json",
    "pred_log2.json"
]
WEIGHTS = [2, 1]

IOU_THR = 0.55
SKIP_BOX_THR = 0.05


def run_wbf():
    all_preds = {}
    for f in FILES:
        if not os.path.exists(f):
            print(f"Warning：Can not find file {f}！")
            return

        with open(f, 'r') as file:
            data = json.load(file)
            valid_data = [
                p for p in data if isinstance(
                    p, dict) and 'bbox' in p and 'image_id' in p]
            all_preds[f] = valid_data
            print(f"Successfully load {f} ({len(valid_data)} inferences)")

    img_ids = set()
    for preds in all_preds.values():
        for p in preds:
            img_ids.add(p['image_id'])

    final_results = []

    for img_id in img_ids:
        boxes_list, scores_list, labels_list = [], [], []

        for f in FILES:
            f_boxes, f_scores, f_labels = [], [], []
            img_preds = [p for p in all_preds[f] if p['image_id'] == img_id]

            for p in img_preds:
                x, y, w, h = p['bbox']

                x1 = min(max(x / 1000.0, 0.0), 1.0)
                y1 = min(max(y / 1000.0, 0.0), 1.0)
                x2 = min(max((x + w) / 1000.0, x1 + 0.001), 1.0)
                y2 = min(max((y + h) / 1000.0, y1 + 0.001), 1.0)

                f_boxes.append([x1, y1, x2, y2])
                f_scores.append(float(p['score']))
                f_labels.append(int(p['category_id']))

            boxes_list.append(f_boxes)
            scores_list.append(f_scores)
            labels_list.append(f_labels)

        if not any(len(b) > 0 for b in boxes_list):
            continue

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=WEIGHTS, iou_thr=IOU_THR, skip_box_thr=SKIP_BOX_THR
        )

        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b[0] * 1000, b[1] * 1000, b[2] * 1000, b[3] * 1000
            final_results.append({
                "image_id": int(img_id),
                "category_id": int(l),
                "bbox": [round(x1, 2),
                         round(y1, 2),
                         round(x2 - x1, 2),
                         round(y2 - y1, 2)],
                "score": round(float(s), 4)
            })

    with open("pred.json", "w") as f:
        json.dump(final_results, f)
    print("\n WBF Complete.")


if __name__ == "__main__":
    run_wbf()
