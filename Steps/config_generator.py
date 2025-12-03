import os
import json
import zipfile
import shutil
from logger.python_logger import ABCLogger

logger = ABCLogger()


def extract_zip(zip_file):
    """
    Extracts the given ZIP file into its parent directory and removes macOS metadata folders.

    This function extracts all contents of a ZIP file to the directory where the ZIP is located.
    macOS often adds a `__MACOSX` metadata folder inside ZIPs; this function automatically
    detects and removes it after extraction.

    Args:
        zip_file (str):
            Absolute or relative path to the ZIP file that needs to be extracted.

    Returns:
        str:
            The folder path where the ZIP contents were extracted (the parent of the ZIP file).

    Side Effects:
        - Creates files/folders extracted from the ZIP.
        - Deletes the `__MACOSX` directory if present.
        - Logs extraction steps and cleanup actions.

    Example:
        >>> extract_zip("/path/to/dataset.zip")
        "/path/to"
    """
    target_folder = os.path.dirname(zip_file)
    logger.info("Extracting ZIP file", zip_file=zip_file, extract_to=target_folder)

    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(target_folder)

    mac_trash = os.path.join(target_folder, "__MACOSX")
    if os.path.exists(mac_trash):
        shutil.rmtree(mac_trash)
        logger.info("Removed __MACOSX folder")

    logger.info("ZIP extracted successfully", extracted_to=target_folder)
    return target_folder


def find_data_folder(root_folder):
    """
    Detects the correct 'data' folder after ZIP extraction.

    Many datasets are zipped with nested folders. This function tries to locate the actual
    `data/` folder by checking:
    - If `data/` exists directly inside `root_folder`
    - If the ZIP extractor created a wrapper directory containing `data/`

    If nothing is found, it returns the root folder itself.

    Args:
        root_folder (str):
            Path where ZIP contents were extracted.

    Returns:
        str:
            Path to the detected `data` folder or the most likely directory containing images.

    Example:
        >>> find_data_folder("/unzipped/dataset")
        "/unzipped/dataset/data"
    """
    direct_data = os.path.join(root_folder, "data")
    if os.path.isdir(direct_data):
        logger.info("Detected 'data' folder", folder=direct_data)
        return direct_data

    subfolders = [
        os.path.join(root_folder, d)
        for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ]

    if len(subfolders) == 1:
        logger.info("Entering wrapper folder", folder=subfolders[0])
        possible_data = os.path.join(subfolders[0], "data")

        if os.path.isdir(possible_data):
            logger.info("Detected 'data' inside wrapper", folder=possible_data)
            return possible_data

        return subfolders[0]

    return root_folder


def get_class_folders(data_folder):
    """
    Scans the data folder and collects logo classes with their associated image paths.

    The expected folder structure is:
        data/
            Class1/
                img1.png
                img2.jpg
            Class2/
                logo.png
            Background/    <-- skipped

    This function:
    - Skips folders named "background"
    - Detects all image files inside each class directory
    - Builds a config entry: { "name": class_name, "logo_paths": [...] }

    Args:
        data_folder (str):
            Path to the folder containing class directories.

    Returns:
        list[dict]:
            List of dictionaries, each containing:
            - name (str): class name
            - logo_paths (list[str]): paths of images inside that class folder

    Example:
        >>> get_class_folders("data/")
        [
            {"name": "ABC", "logo_paths": ["data/ABC/a.png", ...]},
            {"name": "CDE", "logo_paths": ["data/CDE/b.jpg", ...]}
        ]
    """
    classes = []

    for class_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, class_name)

        if not os.path.isdir(folder_path):
            continue

        if class_name.lower() == "background":
            logger.info("Skipping Background folder", folder=class_name)
            continue

        images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not images:
            logger.warning("No logo images found", folder=class_name)
            continue

        classes.append({
            "name": class_name,
            "logo_paths": images
        })

    logger.info("Detected logo classes", total_classes=len(classes))
    return classes


def generate_config(zip_file: str, output_filename: str = "config.json"):
    """
    Generates a config.json file from a dataset ZIP file containing logo class folders.

    Steps performed:
    1. Extract the ZIP
    2. Locate the correct `data/` folder inside extracted content
    3. Identify all logo classes and their image paths
    4. Save output JSON inside the `data/` folder

    The resulting JSON structure:
    {
        "classes": [
            {"name": "ABC", "logo_paths": ["..."]},
            {"name": "CDE", "logo_paths": ["..."]}
        ]
    }

    Args:
        zip_file (str):
            Path to the dataset ZIP file.
        output_filename (str, optional):
            Name of the output config file. Defaults to "config.json".

    Returns:
        str:
            Full path to the generated config file.

    Example:
        >>> generate_config("logos.zip")
        "/unzipped/data/config.json"
    """
    root_after_extract = extract_zip(zip_file)
    data_folder = find_data_folder(root_after_extract)
    class_entries = get_class_folders(data_folder)

    config_path = os.path.join(data_folder, output_filename)
    config = {"classes": class_entries}

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(
        "Configuration file created successfully",
        output_file=config_path,
        class_count=len(class_entries)
    )

    return config_path
