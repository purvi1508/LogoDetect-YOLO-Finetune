# LogoDetect-YOLO-Finetune

Automated pipeline for **brand-specific logo detection** using YOLO. This repository allows you to generate a labeled dataset from your logos and background images, apply advanced augmentations, prepare the dataset for YOLO training, and fine-tune a YOLO model end-to-end.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Approach](#approach)
- [Pipeline](#pipeline)
- [Usage](#usage)

---

## Overview

This project provides a full pipeline for creating a **custom logo detection YOLO model**:

1. Generate dataset configuration from raw logo images.
2. Augment logos on background images, avoiding humans and text regions.
3. Prepare YOLO-compatible train/validation datasets.
4. Create `data.yaml` for YOLO training.
5. Fine-tune YOLO on your logos and export the best model.

---

## Features

- Automated **dataset generation** from zipped logos.
- Advanced **logo augmentation**:
  - Random placement avoiding humans/text.
  - Resizing based on image size.
  - Random transformations (blur, brightness, contrast, RGB shift, gamma, etc.).
- **Flexible dataset preparation** with optional merging and train/val split.
- **YOLO-ready data.yaml** creation.
- End-to-end **YOLO fine-tuning** with optional export formats.
- Logging at every step using a custom logger.

---

## Approach

1. **Data Collection**  
   Collect logos for each brand in separate folders inside a ZIP file.  
   Include a `Background` folder containing background images for augmentation.

2. **Config Generation**  
   Extract the ZIP and generate a `config.json` with all classes and logo paths.

3. **Logo Augmentation**  
   - Detect humans using YOLO and text regions using Tesseract OCR.  
   - Avoid placing logos over these regions.  
   - Apply transformations and randomly place logos (1–4 per image).  
   - Generate YOLO annotation files for each augmented image.

4. **Dataset Preparation**  
   - Optionally merge datasets from multiple sources.  
   - Split the dataset into **train** and **validation** sets in YOLO structure:
     ```
     data/datasets/
       images/train
       images/val
       labels/train
       labels/val
     ```

5. **Create YOLO Data YAML**  
   - Generate a `data.yaml` file containing:
     ```yaml
     train: /absolute/path/to/images/train
     val: /absolute/path/to/images/val
     nc: <number_of_classes>
     names:
       - Brand1
       - Brand2
     ```

6. **YOLO Fine-Tuning**  
   - Fine-tune `yolo11n.pt` on your dataset.  
   - Automatically uses GPU if available.  
   - Optionally export the trained model in formats like `onnx`, `torchscript`, etc.

---

## Pipeline

```text
logos.zip + Backgrounds
      │
      ▼
[config.json] ← generate_config()
      │
      ▼
[Augmented Images + YOLO Annotations] ← run_logo_augmentation()
      │
      ▼
[Dataset Split into Train/Val] ← prepare_dataset()
      │
      ▼
[data.yaml] ← create_data_yaml()
      │
      ▼
[Fine-Tuned YOLO Model] ← train_yolo()
```
## Usage

1. **Prepare Logos and Backgrounds**  
   - Create a ZIP file with the following structure:
```text
data.zip/
  Brand1/
    logo1.png
    logo2.jpg
  Brand2/
    ...
  Background/
    bg1.jpg
    bg2.png
```