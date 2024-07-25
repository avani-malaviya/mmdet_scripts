#!/bin/bash
cd /scratch/disc/a.malaviya/mmdetection
python tools/misc/download_dataset.py --dataset-name balloon --save-dir data --unzip
python mmdet_prepare_balloon_dataset.py
echo !!!!!!!!!!!!!!!!!!!!!!!! training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
python tools/train.py configs/rtmdet/rtmdet_tiny_1xb4-20e_balloon.py
echo !!!!!!!!!!!!!!!!!!!!!!!! infering !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
python mmdet_infer_balloon.py
