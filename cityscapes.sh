cd /scratch/disc/a.malaviya/mmdetection
python mmdet_init_cityscapes.py
echo !!!!!!!!!!!!!!!!!!!!!!!! training !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bash tools/dist_train.sh configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes.py 4 
echo !!!!!!!!!!!!!!!!!!!!!!!! infering !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
python mmdet_infer_cityscapes.py

