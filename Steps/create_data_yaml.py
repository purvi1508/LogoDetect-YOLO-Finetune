import os
import yaml,json
from logger.python_logger import ABCLogger

logger = ABCLogger()

def create_data_yaml(config_path, base_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        classes = config.get("classes", [])
        if not classes:
            raise ValueError("No classes found in config.json")

        class_names = [cls["name"] for cls in classes]
        os.makedirs(base_path, exist_ok=True)
        train_dir = os.path.abspath(os.path.join(base_path, 'images/train'))
        val_dir = os.path.abspath(os.path.join(base_path, 'images/val'))
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        data = {
            'train': train_dir,
            'val': val_dir,
            'nc': len(class_names),
            'names': class_names
        }
        yaml_path = os.path.join(base_path, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info("YAML file created successfully", path=yaml_path, num_classes=len(class_names))
        return yaml_path

    except Exception as e:
        logger.error("Failed to create data.yaml", error=str(e))
        raise

