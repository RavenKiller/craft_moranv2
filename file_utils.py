# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
import math
import matplotlib.pyplot as plt
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

def saveSplitTextRects(img,rot_rects,bboxes,save_folder="result",save_file=True,save_prefix="rect",is_save=True):
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

