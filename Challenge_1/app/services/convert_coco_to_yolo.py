import json
from pathlib import Path
import shutil


def convert_coco_to_yolo():
    # Load COCO annotations
    with open("storage/images/_annotations.coco.json", "r") as f:
        coco = json.load(f)

    Path("yolo_data/images/train").mkdir(parents=True, exist_ok=True)
    Path("yolo_data/images/val").mkdir(parents=True, exist_ok=True)
    Path("yolo_data/labels/train").mkdir(parents=True, exist_ok=True)
    Path("yolo_data/labels/val").mkdir(parents=True, exist_ok=True)

    # image_id to filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    image_annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Split 70/30
    all_images = list(coco["images"])
    split = int(len(all_images) * 0.7)
    train_images = all_images[:split]
    val_images = all_images[split:]

    for img in train_images:
        img_id = img["id"]
        filename = img["file_name"]
        width = img["width"]
        height = img["height"]

        src = f"storage/images/{filename}"
        dst = f"yolo_data/images/train/{filename}"
        if Path(src).exists():
            shutil.copy(src, dst)

        label_path = f"yolo_data/labels/train/{Path(filename).stem}.txt"
        with open(label_path, "w") as f:
            if img_id in image_annotations:
                for ann in image_annotations[img_id]:
                    # COCO bbox format: [x_min, y_min, width, height]
                    x_min, y_min, bbox_w, bbox_h = ann["bbox"]

                    # YOLO format: [x_center, y_center, width, height] (normalized)
                    x_center = (x_min + bbox_w / 2) / width
                    y_center = (y_min + bbox_h / 2) / height
                    w_norm = bbox_w / width
                    h_norm = bbox_h / height

                    # Class 0 (coin)
                    f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

    for img in val_images:
        img_id = img["id"]
        filename = img["file_name"]
        width = img["width"]
        height = img["height"]

        src = f"storage/images/{filename}"
        dst = f"yolo_data/images/val/{filename}"
        if Path(src).exists():
            shutil.copy(src, dst)

        label_path = f"yolo_data/labels/val/{Path(filename).stem}.txt"
        with open(label_path, "w") as f:
            if img_id in image_annotations:
                for ann in image_annotations[img_id]:
                    x_min, y_min, bbox_w, bbox_h = ann["bbox"]
                    x_center = (x_min + bbox_w / 2) / width
                    y_center = (y_min + bbox_h / 2) / height
                    w_norm = bbox_w / width
                    h_norm = bbox_h / height
                    f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

    print(
        f"Converted {len(train_images)} training images and {len(val_images)} validation images"
    )

    # Saving list of training images to prevent data leakage
    train_filenames = [img["file_name"] for img in train_images]
    with open("training_images_list.json", "w") as f:
        json.dump(train_filenames, f)

    return len(train_images), len(val_images)


if __name__ == "__main__":
    convert_coco_to_yolo()
