import os
import os.path as osp
import torch
import torchvision
import numpy as np

# MMDetection
import mmdet
print(mmdet.__version__)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.apis import set_random_seed

# MMCV
import mmcv
from mmcv import Config

import wandb
wandb.login()

def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)

    annotations = []
    images = []
    obj_count = 0

    for idx, v in enumerate(data_infos.values()):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))


        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'balloon'
        }])
    dump(coco_format_json, out_file)



if __name__ == '__main__':

    convert_balloon_to_coco(ann_file='data/balloon/train/via_region_data.json',
                            out_file='data/balloon/train.json',
                            image_prefix='data/balloon/train')
    convert_balloon_to_coco(ann_file='data/balloon/val/via_region_data.json',
                            out_file='data/balloon/val.json',
                            image_prefix='data/balloon/val')

    config_file = 'mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
    cfg = Config.fromfile(config_file)
    
    # Define type and path to the images.
    cfg.dataset_type = 'COCODataset'

    cfg.data.test.ann_file = 'balloon/val/annotation_coco.json'
    cfg.data.test.img_prefix = 'balloon/val/'
    cfg.data.test.classes = ('balloon',)

    cfg.data.train.ann_file = 'balloon/train/annotation_coco.json'
    cfg.data.train.img_prefix = 'balloon/train/'
    cfg.data.train.classes = ('balloon',)

    cfg.data.val.ann_file = 'balloon/val/annotation_coco.json'
    cfg.data.val.img_prefix = 'balloon/val/'
    cfg.data.val.classes = ('balloon',)

    # modify num classes of the model in box head and mask head
    cfg.model.roi_head.bbox_head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes = 1

    # Use the pretrained model.
    cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth' 

 
    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Epochs
    cfg.runner.max_epochs = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    #  Set the checkpoint interval.
    cfg.checkpoint_config.interval = 4

    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'
    cfg.device = 'cuda'
    
    cfg.evaluation.interval = 2
    
    meta = dict()
    meta['balloon_wandb'] = osp.basename(config_file)
    print(meta) 

    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={'project': 'MMDetection-tutorial'},
             interval=10,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=10)]
