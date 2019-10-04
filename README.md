conda create -n dummy python=3.7
conda activate dummy

pip install -r requirements.txt

python iocr.py --in_file_path=receipts/1.jpg --temp_dir=out/