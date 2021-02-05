# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import numpy as np
"""hyper parameters"""
use_cuda = True

from pathlib import Path
import copy

def load_images_path(root):
    images_path = []
    
    for r, d, f in os.walk(root):
        for file in f:
            if file.lower().endswith((".jpg", ".png", ".bmp", ".jpeg")):
                images_path.append(os.path.join(r, file).replace(os.sep, '/'))
            
    return images_path




def print_boxes_cv2(img, boxes, imgfile):
    p = Path(copy.deepcopy(imgfile))
    labelfile = p.with_suffix('.txt')

    if len(boxes) > 0:
        with open(labelfile, 'w') as f:  
            for i in range(len(boxes)):
                box = boxes[i]
                
                cx = (box[0] + box[2]) / 2.
                cy = (box[1] + box[3]) / 2.
                w = box[2] - box[0]
                h = box[3] - box[1]
                
                cx = np.clip(cx, 0, 1.)
                cy = np.clip(cy, 0, 1.)
                w = np.clip(w, 0, 1.)
                h = np.clip(h, 0, 1.)
                
                cls_conf = box[5]
                cls_id = box[6]
                f.write(f'{cls_id:d} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

def detect_cv2(m, imgfile):
    import cv2
    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        print(boxes)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

#    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)
    print_boxes_cv2(img, boxes[0], imgfile)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfolder', type=str,
                        help='path of your image folder.', dest='imgfolder')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.imgfolder:
        m = Darknet(args.cfgfile)

        m.print_network()
        m.load_weights(args.weightfile)
        print('Loading weights from %s... Done!' % (args.weightfile))

        if use_cuda:
            m.cuda()

        imgfiles = load_images_path(args.imgfolder)
        for imgfile in imgfiles:
            detect_cv2(m, imgfile)
