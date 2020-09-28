
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import tools.imgproc as imgproc
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

class strLabelConverterForAttention(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, sep):
        self._scanned_list = False
        self._out_of_list = ''
        self._ignore_case = True
        self.sep = sep
        self.alphabet = alphabet.split(sep)

        self.dict = {}
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

    def scan(self, text):
        # print(text)
        text_tmp = text
        text = []
        for i in range(len(text_tmp)):
            text_result = ''
            for j in range(len(text_tmp[i])):
                chara = text_tmp[i][j].lower() if self._ignore_case else text_tmp[i][j]
                if chara not in self.alphabet:
                    if chara in self._out_of_list:
                        continue
                    else:
                        self._out_of_list += chara
                        file_out_of_list = open("out_of_list.txt", "a+")
                        file_out_of_list.write(chara + "\n")
                        file_out_of_list.close()
                        print('" %s " is not in alphabet...' % chara)
                        continue
                else:
                    text_result += chara
            text.append(text_result)
        text_result = tuple(text)
        self._scanned_list = True
        return text_result

    def encode(self, text, scanned=True):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        self._scanned_list = scanned
        if not self._scanned_list:
            text = self.scan(text)

        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            return ''.join([self.alphabet[i] for i in t])
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l])))
                index += l
            return texts

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def loadData(v, data):
    major, _ = get_torch_version()

    if major >= 1:
        v.resize_(data.size()).copy_(data)
    else:
        v.data.resize_(data.size()).copy_(data)

def get_torch_version():
    """
    Find pytorch version and return it as integers
    for major and minor versions
    """
    torch_version = str(torch.__version__).split(".")
    return int(torch_version[0]), int(torch_version[1])

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', save_file=True, verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "/res_" + filename + '.txt'
        res_img_file = dirname + "/res_" + filename + '.jpg'
        
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        if save_file:
            cv2.imwrite(res_img_file, img)
        return img

def saveSplitTextRects(img,rot_rects,save_folder="result",save_file=True,save_prefix="rect",is_save=True):
    loose = 5 # add more pixels into the rectangle
    img_cuts = []
    for k,rectangle in enumerate(rot_rects):
        cx = rectangle[0][0]
        cy = rectangle[0][1]
        width = rectangle[1][0]
        height = rectangle[1][1]
        angle = rectangle[2]
        (h,w) = img.shape[:2]
        if abs(angle)>1:
            #设置旋转矩阵
            M = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
            cos = np.abs(M[0,0])
            sin = np.abs(M[0,1])
            
            # 计算图像旋转后的新边界
            nW = int((h*sin)+(w*cos))
            nH = int((h*cos)+(w*sin))
            # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
            M[0,2] += (nW/2) - w/2
            M[1,2] += (nH/2) - h/2
            
            img_rotated = cv2.warpAffine(img,M,(nW,nH))

            res = np.dot(M, np.array([[cx], [cy], [1]]))
            cx = int(res[0,0])
            cy = int(res[1,0])
            x1 = int(cx-width/2)
            y1 = int(cy-height/2)
            x2 = int(cx+width/2)
            y2 = int(cy-height/2)
            x3 = int(cx+width/2)
            y3 = int(cy+height/2)
            x4 = int(cx-width/2)
            y4 = int(cy+height/2)
            img_cut = img_rotated[max(0,y1-loose):min(nH,y3+loose),max(0,x1-loose):min(nW,x3+loose),:]
        else:
            x1 = int(cx-width/2)
            y1 = int(cy-height/2)
            x3 = int(cx+width/2)
            y3 = int(cy+height/2)
            img_cut = img[max(0,y1-loose):min(h,y3+loose),max(0,x1-loose):min(w,x3+loose),:]
        m,n,_ = img_cut.shape
        if m>n+10:
            img_cut = img_cut.transpose(1,0,2)
            img_cut = img_cut[::-1,:,:]
        img_cuts.append(img_cut)
        if save_file:
            cv2.imwrite(os.path.join(save_folder,save_prefix+"_"+str(k)+".jpg"),img_cut[:,:,::-1])
    return img_cuts


