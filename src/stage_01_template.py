import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories,extract_data
from src.utils.Data_mgnt import validation_image
import random


STAGE = "GET DATA" ## <<< change stage name 

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
    bad_data=config['data']['bad_data']
  
    data_file=os.path.join(Local_dir,'data.zip')
    create_directories([Local_dir])
    if not os.path.isfile(data_file):
        shutil.move("E:\Machine Learning Projects\MlFlow-CNN\data.zip",Local_dir)
        logging.info(f"data.zip moved to {Local_dir} folder")
    else:
        logging.info(f"data.zip already present at {Local_dir} folder")

    if not os.path.exists(os.path.join(Local_dir,"PetImages")):
        extract_data(data_file,Local_dir)
    else:
        logging.info(f'{data_file} already extracted')
    
    create_directories([os.path.join(Local_dir,"bad_dir")])

    validation_image(os.path.join(Local_dir,'PetImages'),os.path.join(Local_dir,'bad_dir'))


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