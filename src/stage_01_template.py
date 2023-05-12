import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
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
  
    data_file=os.path.join(Local_dir,'data.zip')
    create_directories([Local_dir])
    if not os.path.isfile(data_file):
        shutil.move("E:\Machine Learning Projects\MlFlow-CNN\data.zip",Local_dir)
        logging.info(f"data.zip moved to {Local_dir} folder")
    else:
        logging.info(f"data.zip already present at {Local_dir} folder")
    pass


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