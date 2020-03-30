"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import json
import zipfile

import tools.utils as utils
import tools.dataset as dataset
import tools.imgproc as imgproc
import tools.craft_utils as craft_utils
from models.craft import CRAFT
from models.moran import MORAN

import matplotlib.pyplot as plt
from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
    
def craft_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    tmp1 = score_link.copy()
    tmp2 = score_text.copy()

    # Post-processing
    boxes, polys, rot_rects = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    rot_rects = craft_utils.adjustResultCoordinatesNew(rot_rects, ratio_w, ratio_h)
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, ret_score_text,rot_rects

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
# CRAFT args
parser.add_argument('--craft_trained_model', default='pretrained/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--img_path', default='test/1.jpg', type=str, help='folder path to input images')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='pretrained/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
# moran 

parser.add_argument('--moran_path', default='pretrained/moran.pth', type=str, help='pretrained moran model')
args = parser.parse_args()
moran_path = args.moran_path
alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'

if __name__ == '__main__':
    ################################################
    # cv2 initialize
    ################################################
    cap = cv2.VideoCapture(0)
    ################################################
    # CRAFT loading part
    ################################################
    # load net
    net = CRAFT()     # initialize
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.craft_trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.craft_trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    ################################################
    # MORAN loading part
    ################################################
    cuda_flag = False
    if torch.cuda.is_available():
        cuda_flag = True
        MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=cuda_flag)
        MORAN = MORAN.cuda()
    else:
        MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=cuda_flag)

    print('loading pretrained model from %s' % moran_path)
    if cuda_flag:
        state_dict = torch.load(moran_path)
    else:
        state_dict = torch.load(moran_path, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        MORAN_state_dict_rename[name] = v
    MORAN.load_state_dict(MORAN_state_dict_rename)

    for p in MORAN.parameters():
        p.requires_grad = False
    MORAN.eval()
    while(cap.isOpened()):
        all_text = []
        all_text_reverse = []

        ################################################
        # CRAFT processing part
        ################################################
        # load data
        
        tik = time.time()
        ret, image = cap.read()
        # image = cv2.imread('test/1.jpg')
        image_raw = image.copy()
        bboxes, score_text,rot_rects = craft_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)
        print("time1: ",time.time()-tik)
        # save text rectangles
        filename, file_ext = os.path.splitext(os.path.basename(args.img_path))
        # 这个可以保存切分的图片
        img_cuts = utils.saveSplitTextRects(image,rot_rects,save_file=False,save_prefix="rect_"+filename)
        print("time2: ",time.time()-tik)
        if not img_cuts:
            cv2.imshow('Capture', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        ###############################################
        # MORAN processing part
        ################################################
        converter = utils.strLabelConverterForAttention(alphabet, ':')
        transformer = dataset.resizeNormalize((100, 32))
        images = [transformer(Image.fromarray(img.astype('uint8')).convert('L')) for img in img_cuts]
        images = [Variable(img.view(1, *img.size())) for img in images]
        all_image = torch.cat(images,axis=0)
        if cuda_flag:
            all_image = all_image.cuda()
        text = torch.LongTensor(1 * 5)
        length = torch.IntTensor(1)
        text = Variable(text)
        length = Variable(length)

        # 从单张修改为多张，只需要改Length
        # 作者给的处理工具已经考虑了多个图片同时处理的情况
        max_iter = 20
        t, l = converter.encode('0'*max_iter)
        utils.loadData(text, t)
        utils.loadData(length, l)
        length = torch.ones(len(img_cuts))*20
        length = length.int()
        output = MORAN(all_image, length, text, text, test=True, debug=False)
        preds, preds_reverse = output[0]
        _, preds = preds.max(1)
        _, preds_reverse = preds_reverse.max(1)

        sim_preds = converter.decode(preds.data, length.data)
        all_text = [v.strip().split('$')[0] for v in sim_preds]
        print(sim_preds)
        print("time3: ",time.time()-tik)
        result_img = utils.saveResult(args.img_path, image_raw[:,:,::-1], bboxes,save_file=False, texts=all_text)
        print("time4: ",time.time()-tik)
        print(all_text)
        cv2.imshow('Capture', result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break