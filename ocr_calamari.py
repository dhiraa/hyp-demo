'''
Demo sample example of how to include calamari_ocr into python code
'''

import sys
import os
from glob import glob

sys.path.append(os.getcwd())

from calamari_ocr.ocr.datasets import DataSetType
from calamari_ocr.scripts.predict import run as calamari_ocr_run

calamari_models = ['models/calamari_models/model_00117200.ckpt',
                   'models/calamari_models/model_00132600.ckpt',
                   'models/calamari_models/model_00009800.ckpt']

class CalamariArgs:
    batch_size = 1
    checkpoint = calamari_models
    dataset = DataSetType.FILE
    extended_prediction_data = False
    extended_prediction_data_format = 'json'
    files = []
    no_progress_bars = False
    output_dir = None
    pagexml_text_index = 1
    processes = 1
    text_files = None
    verbose = False
    voter = 'confidence_voter_default_ctc'
    extension = None

def find_files_with_ext(search_folder, exts=['.JPG', '.jpg', '.png']):
    all_files = glob(search_folder + '**/**', recursive=True)
    bag = []
    if exts:
        for _ext in exts:
            bag += [file for file in all_files if file.endswith(_ext)]
    else:
        bag = all_files
    return bag

def get_all_input_files(source_dir, input_files_types=['.JPG', '.jpg', '.png']):
    """Get the list of images files from the source directory"""
    return find_files_with_ext(source_dir, input_files_types)

def calamari(source_dir, destination_dir):
    """Plugin module should implement this to handle all the files in the given directory"""

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    in_files = get_all_input_files(source_dir=source_dir)

    print(">>>>>>>>>>>>>>>>>>>")
    print(in_files)

    CalamariArgs.files = in_files
    CalamariArgs.output_dir = destination_dir
    CalamariArgs.batch_size = len(in_files)
    CalamariArgs.processes = 4
    calamari_ocr_run(CalamariArgs)

