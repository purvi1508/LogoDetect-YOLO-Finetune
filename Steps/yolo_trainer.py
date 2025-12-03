import os
from ultralytics import YOLO
from logger.python_logger import ABCLogger
import torch

logger = ABCLogger()

def train_yolo(export_formats=None):
    try:
        logger.info("Starting YOLO model training...")

        model = YOLO("yolo11n.pt")
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        results = model.train(
            data="data/datasets/data.yaml",
            epochs=50,
            imgsz=480,
            device=device_type,
            batch=4,
            amp=True
        )

        logger.info("YOLO model training complete.")

        # Export trained model to optional formats
        if export_formats:
            for fmt in export_formats:
                logger.info(f"Exporting model to {fmt}...")
                model.export(format=fmt)
            logger.info("Model export complete.")

        # Return path to best weights
        best_weights = results.best
        logger.info(f"Best weights saved at: {best_weights}")
        return best_weights

    except Exception as e:
        logger.error("Error occurred during YOLO training or export", error=str(e))
        raise
