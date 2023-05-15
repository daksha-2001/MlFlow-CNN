import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.models import log_model_summary
import random
import tensorflow as tf


STAGE = "Base_Model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    input_shape=config['model']['input_image']
    logging.info("Layers defined")
    LAYERS = [
    tf.keras.layers.Input(shape=tuple(input_shape)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
    ]

    classifier = tf.keras.Sequential(LAYERS)

    logging.info(f'Base model summary: \n {log_model_summary(classifier)}')

    classifier.compile(
    optimizer=tf.keras.optimizers.Adam(config['model']['lr']),
    loss=config['model']['loss'],
    metrics=config['model']['metric'])

    path_to_model_dir=os.path.join(
        config['data']['local_dir'],config['data']['model_dir']
    )
    create_directories([path_to_model_dir])
    
    path_to_model=os.path.join(path_to_model_dir,config['data']['initial_model_file']
    )
    classifier.save(path_to_model)

    logging.info(f'Model is saved at {path_to_model}')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e