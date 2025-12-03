from logger.python_logger import ABCLogger
from dotenv import load_dotenv
logger = ABCLogger()


from Steps.config_generator import generate_config

zip_path = "data.zip"           # user provides this
output_config = "config.json"   # generate here

generate_config(zip_path, output_config)

print("Config generated!")
from Steps.logo_augmentor import run_logo_augmentation
run_logo_augmentation()
from Steps.dataset_preparation import prepare_dataset
prepare_dataset()
from Steps.create_data_yaml import create_data_yaml
config_path = "logo_fine_tuning/data/config.json"
base_path = "data/datasets"
yaml_path = create_data_yaml(config_path, base_path)
print("YOLO data.yaml created at:", yaml_path)

from Steps.yolo_trainer import train_yolo
train_yolo()