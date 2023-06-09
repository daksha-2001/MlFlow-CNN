import shutil
import imghdr
from PIL import Image
import os
import logging

def validation_image(PARENT_DIR,BAD_DATA_DIR):

    for dirs in os.listdir(PARENT_DIR):
        full_path_data_dir = os.path.join(PARENT_DIR, dirs)
        for imgs in os.listdir(full_path_data_dir):
            path_to_img = os.path.join(full_path_data_dir, imgs)
            try:
                img = Image.open(path_to_img)
                img.verify()
                    
                if len(img.getbands()) !=3 or imghdr.what(path_to_img) not in ['jpeg','png']:
                    bad_data_path = os.path.join(BAD_DATA_DIR, imgs)
                    shutil.move(path_to_img, bad_data_path)
                    logging.info(f"{path_to_img} is not in correct format! hence moving to {bad_data_path}")
                    continue
                    
                
            except Exception as e:

                bad_data_path = os.path.join(BAD_DATA_DIR, imgs)
                shutil.move(path_to_img, bad_data_path)
                logging.info(f"{path_to_img} is bad! hence moving to {bad_data_path}")

        logging.info(f"All images are verifed successfully")