import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf


STAGE = "Training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    Local_dir=config['data']['local_dir']

    logging.info(f'Reading dataset from {Local_dir} directory')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(Local_dir,'PetImages'),
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=tuple(config['model']['input_image'])[:-1],
    batch_size=32)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(Local_dir,'PetImages'),
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=tuple(config['model']['input_image'])[:-1],
    batch_size=32
)
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    path_to_model_dir=os.path.join(
        config['data']['local_dir'],config['data']['model_dir']
    )
    path_to_model_ini=os.path.join(path_to_model_dir,config['data']['initial_model_file']
    )
    path_to_model_final=os.path.join(path_to_model_dir,config['data']['final_model_file']
    )
    
    logging.info(f'Loading {path_to_model_ini} model')
    classifier=tf.keras.models.load_model(path_to_model_ini)

    logging.info(f'Training started')
    classifier.fit(train_ds,epochs=1,validation_data=val_ds)

    logging.info(f'saving {path_to_model_final} model')
    classifier.save(path_to_model_final)


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