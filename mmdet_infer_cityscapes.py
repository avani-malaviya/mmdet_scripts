from mmdet.apis import DetInferencer
import glob

# Choose to use a config
config = 'configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes.py'
# Setup a checkpoint file to load
checkpoint = './work_dirs/faster-rcnn_r50_fpn_1x_cityscapes/epoch_8.pth'

# Set the device to be used for evaluation
device = 'cuda:0'

# Initialize the DetInferencer
inferencer = DetInferencer(config, checkpoint, device)

# Use the detector to do inference
img = './data/cityscapes/leftImg8bit/test/bonn/bonn_000010_000019_leftImg8bit.png'
result = inferencer(img, out_dir='./output')
