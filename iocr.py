import os
import shutil
import argparse
import sys
# Appending vitaFlow main Path
import warnings
from absl import logging
from tqdm import tqdm
from crop_to_box import crop_to_box
from ocr_calamari import calamari, get_all_input_files

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity(logging.ERROR)

sys.path.append(os.path.abspath('.'))
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import time

import cv2
import numpy as np
import fire
import tensorflow as tf
from prediction import resize_image, sort_poly, detect
from grpc_predict import get_text_segmentation_pb, read_image

tf.get_logger().setLevel(logging.ERROR)

OUT_DIR = "/tmp/oct/"

def east(image_file_path, out_dir, model_dir="models/vf_east_models/east/EASTIEstimatorModel/exported/1558013588/"):
    print(image_file_path)
    predict_fn = tf.contrib.predictor.from_saved_model(model_dir)
    im, img_resized, ratio_h, ratio_w = read_image(image_file_path)
    result = predict_fn({'images': img_resized})

    get_text_segmentation_pb(img_mat=im,
                             result=result,
                             output_dir=out_dir,
                             file_name=os.path.basename(image_file_path),
                             ratio_h=ratio_h,
                             ratio_w=ratio_w)

def get_text_file(pred_text_dir, out_file_path):
    lines = []

    in_files = get_all_input_files(source_dir=pred_text_dir,
                                   input_files_types=[".txt"])


    for each_in_text_prediction in tqdm(in_files):
        if os.path.isfile(out_file_path):
            #  read the file and append values
            with open(out_file_path, "a") as fd:
                line = open(each_in_text_prediction, "r").read()
                lines.append(line)
                fd.write(line)
                fd.write("\n")
        else:
            # create a new file with headers
            with open(out_file_path, "w") as fd:
                line = open(each_in_text_prediction, "r").read()
                lines.append(line)
                fd.write(line)
                fd.write("\n")

    return lines

def ocr(in_file_path, temp_dir="/tmp/ocr/"):
    """

    :param in_file_path:
    :return:
    """
    global OUT_DIR
    OUT_DIR = temp_dir

    file_name = os.path.basename(in_file_path)
    file_name_base = str(file_name.split(".")[0])

    east(in_file_path, out_dir=OUT_DIR + file_name_base + "/east")

    crop_to_box(gt_text_file_loc=OUT_DIR + file_name_base + "/east/" + file_name_base + ".txt",
                source_image_loc=OUT_DIR + file_name_base + "/east/" + file_name,
                cropped_dir=OUT_DIR + file_name_base + "/cropped/")

    calamari(source_dir=OUT_DIR + file_name_base + "/cropped/",
             destination_dir=OUT_DIR + file_name_base + "/text")

    ret = get_text_file(pred_text_dir=OUT_DIR + file_name_base + "/text",
                        out_file_path=OUT_DIR + file_name_base + ".txt")

    print(ret)
    return ret

if __name__ == '__main__':
    fire.Fire(ocr)



