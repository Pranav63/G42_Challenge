from ultralytics import YOLO
import yaml
from pathlib import Path


def train():
    # Create YOLO config
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
    model = YOLO("yolov8s.pt")  # Use small model (better than nano)

    results = model.train(
        data="yolo_data/data.yaml",
        epochs=100,  # More epochs since we have good data
        imgsz=640,
        batch=16,
        patience=20,
    )

    # Save best model
    model.save("coin_model_final.pt")
    print("Training complete!")


if __name__ == "__main__":
    # First convert annotations
    from convert_coco_to_yolo import convert_coco_to_yolo

    convert_coco_to_yolo()

    # Then train
    train()
