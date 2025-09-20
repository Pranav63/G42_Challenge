from ultralytics import YOLO
import yaml
from pathlib import Path
from app.services.convert_coco_to_yolo import convert_coco_to_yolo


def train():
    data_config = {
        "path": str(Path("yolo_data").absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["coin"],
    }

    with open("yolo_data/data.yaml", "w") as f:
        yaml.dump(data_config, f)

    # Train model
    model = YOLO("yolov8s.pt")

    model.train(
        data="yolo_data/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        patience=20,
    )

    model.save("coin_model_final.pt")
    print("Training complete!")


if __name__ == "__main__":

    convert_coco_to_yolo()

    train()
