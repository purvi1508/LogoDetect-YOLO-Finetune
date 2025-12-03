import shutil
import os
import random
import json
from logger.python_logger import ABCLogger

logger = ABCLogger()

def read_user_input(json_file_path='my_input.json'):
    try:
        if not os.path.exists(json_file_path):
            logger.error("JSON config file not found", path=json_file_path)
            raise FileNotFoundError(f"{json_file_path} not found")

        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            logger.info("Loaded user input JSON", file=json_file_path)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format", file=json_file_path)
            raise

        raw_response = data.get("merge_datasets", {}).get("user_input", None)
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        if isinstance(raw_response, bool):
            user_answer = raw_response
        elif isinstance(raw_response, str):
            if raw_response.lower() == "true":
                user_answer = True
            elif raw_response.lower() == "false":
                user_answer = False
            else:
                raise ValueError("Invalid string for 'user_input'. Use 'true' or 'false'.")
        else:
            raise ValueError("Invalid type for 'user_input'. Must be a boolean or 'true'/'false' string.")
        return user_answer
    except Exception as e:
        print(f"Error reading user input: {e}")
        return None

def merge_folders(source_folder, destination_folder):
    if os.path.abspath(source_folder) == os.path.abspath(destination_folder):
        raise ValueError("Source and destination folders must be different.")
    if not os.path.isdir(source_folder) or not os.path.isdir(destination_folder):
        raise FileNotFoundError("Both paths must be valid directories.")
    
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isfile(source_path):
            shutil.move(source_path, destination_path)
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)

def split_data(source_dir, output_dir, train_ratio=0.8):
    image_train_dir = os.path.join(output_dir, 'images/train')
    image_val_dir = os.path.join(output_dir, 'images/val')
    label_train_dir = os.path.join(output_dir, 'labels/train')
    label_val_dir = os.path.join(output_dir, 'labels/val')

    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    all_files = os.listdir(source_dir)
    images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    paired_files = []
    for image in images:
        base_name = os.path.splitext(image)[0]
        label_file = base_name + '.txt'
        if label_file in all_files:
            paired_files.append((image, label_file))

    if not paired_files:
        print("No image-label pairs found.")
        return

    random.shuffle(paired_files)
    split_idx = int(len(paired_files) * train_ratio)
    train_files = paired_files[:split_idx]
    val_files = paired_files[split_idx:]

    def copy_files(file_list, img_dest, label_dest):
        for img_file, label_file in file_list:
            shutil.copy(os.path.join(source_dir, img_file), os.path.join(img_dest, img_file))
            shutil.copy(os.path.join(source_dir, label_file), os.path.join(label_dest, label_file))

    copy_files(train_files, image_train_dir, label_train_dir)
    copy_files(val_files, image_val_dir, label_val_dir)

    print(f"Dataset split completed:\n  Train: {len(train_files)} pairs\n  Validation: {len(val_files)} pairs")


def prepare_dataset():
    user_merge_choice = read_user_input()

    if user_merge_choice is True:
        final_directory = 'data/final'
        directories = [d for d in os.listdir(final_directory)
                       if os.path.isdir(os.path.join(final_directory, d)) and d != '__MACOSX']

        if directories:
            source_folder = os.path.join(final_directory, directories[0])
            destination_folder = "data/output_images_with_logos"
            os.makedirs(destination_folder, exist_ok=True)
            merge_folders(source_folder, destination_folder)
        else:
            print("No valid directories found inside 'data/final'.")

    split_data("data/output_images_with_logos", "data/datasets")

