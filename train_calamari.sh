python icdar/calamari_dataset.py --in_path=/opt/vlab/icdar-2019-data/train/
python icdar/calamari_dataset.py --in_path=/opt/vlab/icdar-2019-data/test/
calamari-train --files cropped/*.png --num_threads=8 --batch_size=64
