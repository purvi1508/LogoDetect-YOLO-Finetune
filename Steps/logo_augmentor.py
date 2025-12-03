import os
import json
import cv2
import numpy as np
import pytesseract
import random
from PIL import ImageEnhance, Image
from pytesseract import Output
from scipy.spatial import cKDTree
import numpy as np
from logger.python_logger import ABCLogger
import shutil
from itertools import cycle

tesseract_path = shutil.which("tesseract")
logger = ABCLogger()


def build_adjacency_kdtree(bboxes, threshold=100):
    if not bboxes:
        return {}

    centers = np.array([[(x + w/2), (y + h/2)] for x, y, w, h in bboxes])
    tree = cKDTree(centers)
    adjacency_list = {i: [] for i in range(len(bboxes))}

    for i, center in enumerate(centers):
        neighbors = tree.query_ball_point(center, r=threshold)
        adjacency_list[i] = [j for j in neighbors if j != i]

    return adjacency_list

def are_boxes_nearby(bbox1, bbox2, threshold=100):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 + w1 < x2 - threshold or x2 + w2 < x1 - threshold or y1 + h1 < y2 - threshold or y2 + h2 < y1 - threshold)

def dfs(node, visited, adjacency_list, component):
    visited[node] = True
    component.append(node)
    for neighbor in adjacency_list[node]:
        if not visited[neighbor]:
            dfs(neighbor, visited, adjacency_list, component)

def divide_free_space(free_space, num_logos):
    x, y, w, h = free_space
    sub_spaces = []
    for _ in range(num_logos):
        sub_width = random.randint(w // num_logos, w // num_logos + w // 10)
        sub_height = random.randint(h // num_logos, h // num_logos + h // 10)
        sub_width = min(sub_width, w)
        sub_height = min(sub_height, h)
        sub_x = random.randint(x, x + w - sub_width)
        sub_y = random.randint(y, y + h - sub_height)
        sub_spaces.append((sub_x, sub_y, sub_width, sub_height))
    return sub_spaces

def combine_bboxes(bboxes, component):
    min_x = min([bboxes[i][0] for i in component])
    min_y = min([bboxes[i][1] for i in component])
    max_x = max([bboxes[i][0] + bboxes[i][2] for i in component])
    max_y = max([bboxes[i][1] + bboxes[i][3] for i in component])
    return (min_x, min_y, max_x - min_x, max_y - min_y)

def is_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    return True

def detect_humans(image):
    """
    Returns bounding boxes of humans detected in the image.
    Format: (x, y, w, h)
    """
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt") 

    results = model(image)[0]
    boxes = []

    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = det
        if int(cls) == 0: 
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return boxes


def resize_logo(logo, max_width, max_height, image_width, image_height,
                ratio_min=0.08, ratio_max=0.1):
    """
    Resize logo based on:
    1. Random scale factor
    2. Max allowed bounding-box size (max_width/max_height)
    3. Desired logo-to-image area ratio
    """

    logo_h, logo_w = logo.shape[:2]
    img_area = image_width * image_height
    target_ratio = random.uniform(ratio_min, ratio_max)
    target_logo_area = img_area * target_ratio
    aspect_ratio = logo_w / logo_h
    new_height = int((target_logo_area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    if new_width > max_width:
        scale = max_width / new_width
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)

    if new_height > max_height:
        scale = max_height / new_height
        new_height = int(new_height * scale)
        new_width = int(new_width * scale)
    resized = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def overlay_logo(image, logo, x, y):
    logo_height, logo_width = logo.shape[:2]
    if logo.shape[2] == 4:  # PNG with alpha
        logo_bgr = logo[:, :, :3]
        alpha_channel = logo[:, :, 3]
        image_section = image[y:y + logo_height, x:x + logo_width]
        _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        image_section_bg = cv2.bitwise_and(image_section, image_section, mask=mask_inv)
        logo_fg = cv2.bitwise_and(logo_bgr, logo_bgr, mask=mask)
        combined = cv2.add(image_section_bg, logo_fg)
        image[y:y + logo_height, x:x + logo_width] = combined
    else:
        image[y:y + logo_height, x:x + logo_width] = logo
    return image

# ========================= Transformations ========================= #

def apply_rgb_shift(image):
    shift = np.random.randint(-10, 10, 3)
    image = cv2.add(image, shift)
    return image

def apply_hue_saturation_value(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = np.random.randint(0, 10)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_channel_shuffle(image):
    channels = list(cv2.split(image))
    random.shuffle(channels)
    return cv2.merge(channels)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def apply_random_contrast(image):
    enhancer = ImageEnhance.Contrast(Image.fromarray(image))
    return np.array(enhancer.enhance(random.uniform(0.5, 1.5)))

def apply_random_gamma(image):
    gamma = random.uniform(1.0, 2.0)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_random_brightness(image):
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))
    return np.array(enhancer.enhance(random.uniform(0.5, 1.5)))

def apply_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def apply_median_blur(image):
    return cv2.medianBlur(image, 3)

def apply_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_jpeg_compression(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 90)]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

transforms = {
    "RGB Shift": apply_rgb_shift,
    "HueSaturationValue": apply_hue_saturation_value,
    "ChannelShuffle": apply_channel_shuffle,
    "CLAHE": apply_clahe,
    "RandomContrast": apply_random_contrast,
    "RandomGamma": apply_random_gamma,
    "RandomBrightness": apply_random_brightness,
    "Blur": apply_blur,
    "MedianBlur": apply_median_blur,
    "ToGray": apply_to_gray,
    "JpegCompression": apply_jpeg_compression
}

# --- Configuration Functions ---
def is_transformation_enabled(transform_name, transform_config):
    augmentation_settings = transform_config.get("augmentation_settings", {})
    transform_details = augmentation_settings.get(transform_name, {})
    return transform_details.get("enabled", False)

def get_enabled_transforms(transform_config):
    return [name for name in transforms.keys() if is_transformation_enabled(name, transform_config)]

# --- Main Processing Function ---
def process_images(config, transform_config, image_path, output_folder, tot_images=500):
    os.makedirs(output_folder, exist_ok=True)
    classes = config["classes"]
    logo_paths = {cls["name"]: cls["logo_paths"] for cls in classes}

    logger.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Unable to read image: {image_path}")
        return

    enabled_transforms = get_enabled_transforms(transform_config)
    for transform_name in enabled_transforms:
        logger.debug(f"Applying transformation: {transform_name}")
        image = transforms[transform_name](image)

    image_height, image_width = image.shape[:2]
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    detection_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    ocr_bboxes = [(detection_data['left'][i], detection_data['top'][i],
               detection_data['width'][i], detection_data['height'][i])
              for i in range(len(detection_data['text'])) if detection_data['text'][i].strip()]
    human_bboxes = detect_humans(image)
    bboxes = ocr_bboxes + human_bboxes
    adjacency_list = build_adjacency_kdtree(bboxes, threshold=100)
    visited = [False] * len(bboxes)
    components = []
    for i in range(len(bboxes)):
        if not visited[i]:
            component = []
            dfs(i, visited, adjacency_list, component)
            components.append(component)
    combined_bboxes = [combine_bboxes(bboxes, component) for component in components]
    combined_bboxes = sorted(combined_bboxes, key=lambda x: x[1])


    free_spaces = []
    last_y = 0
    for bbox in combined_bboxes:
        x, y, w, h = bbox
        if y > last_y:
            free_spaces.append((0, last_y, image_width, y - last_y))
        last_y = y + h
    if last_y < image_height:
        free_spaces.append((0, last_y, image_width, image_height - last_y))
    class_usage = {cls["name"]: 0 for cls in classes}
    logo_usage = {cls["name"]: {path: 0 for path in cls["logo_paths"]} for cls in classes}
    for i in range(tot_images):
        output_image = image.copy()
        num_logos = random.randint(1, 4)

        if not bboxes:
            free_space = [(0, 0, image.shape[1], image.shape[0])]
            free_spaces = divide_free_space(free_space[0], num_logos)

        selected_free_spaces = random.sample(free_spaces, min(len(free_spaces), num_logos))
        annotations = []
        placed_logos = []
        for free_space in selected_free_spaces:
            x, y, w, h = free_space
            selected_class_name = min(class_usage, key=class_usage.get)
            selected_class = next(cls for cls in classes if cls["name"] == selected_class_name)
            class_usage[selected_class_name] += 1
            min_usage_logo = min(logo_usage[selected_class_name], key=logo_usage[selected_class_name].get)
            logo_usage[selected_class_name][min_usage_logo] += 1
            selected_logo_path = min_usage_logo

            logo = cv2.imread(selected_logo_path, cv2.IMREAD_UNCHANGED)

            if logo is None:
                logger.warning(f"Failed to load logo: {selected_logo_path}")
                continue

            resized_logo = resize_logo(logo, w, h,image_width, image_height)
            logo_height, logo_width = resized_logo.shape[:2]

            if w >= logo_width and h >= logo_height:
                for _ in range(10):
                    offset_x = random.randint(0, w - logo_width)
                    offset_y = random.randint(0, h - logo_height)
                    new_bbox = (x + offset_x, y + offset_y, logo_width, logo_height)
                    if all(not is_overlap(new_bbox, placed) for placed in placed_logos):
                        overlay_logo(output_image, resized_logo, x + offset_x, y + offset_y)
                        x_center = (x + offset_x + logo_width / 2) / image_width
                        y_center = (y + offset_y + logo_height / 2) / image_height
                        norm_width = logo_width / image_width
                        norm_height = logo_height / image_height
                        annotations.append(f"{classes.index(selected_class)} {x_center} {y_center} {norm_width} {norm_height}")
                        placed_logos.append(new_bbox)
                        break

        if annotations:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_folder, f'{base_name}_logo_{i + 1}.jpg')
            annotation_path = os.path.join(output_folder, f'{base_name}_logo_{i + 1}.txt')
            cv2.imwrite(output_path, output_image)
            with open(annotation_path, 'w') as f:
                f.write('\n'.join(annotations))

    logger.info(f"Finished processing image: {image_path}")

# --- Entry Point ---
def run_logo_augmentation():
    config_path = 'data/config.json'
    transform_config_path = 'my_input.json'
    background_folder = 'data/Background'
    image_folder = 'data/output_images_with_logos'
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')

    logger.info("Loading configurations...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(transform_config_path, 'r') as f:
        transform_config = json.load(f)

    logger.info("Starting image augmentation...")
    image_files = [f for f in os.listdir(background_folder) if f.lower().endswith(image_extensions)]
    for image_file in image_files:
        image_path = os.path.join(background_folder, image_file)
        process_images(config, transform_config, image_path, image_folder)

    logger.info("Image augmentation completed.")

