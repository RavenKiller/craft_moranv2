
#-*- coding: utf-8 -*-
"""
  印刷文字识别WebAPI接口调用示例接口文档(必看)：https://doc.xfyun.cn/rest_api/%E5%8D%B0%E5%88%B7%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB.html
  上传图片base64编码后进行urlencode要求base64编码和urlencode后大小不超过4M最短边至少15px，最长边最大4096px支持jpg/png/bmp格式
  (Very Important)创建完webapi应用添加合成服务之后一定要设置ip白名单，找到控制台--我的应用--设置ip白名单，如何设置参考：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=41891
  错误码链接：https://www.xfyun.cn/document/error-code (code返回错误码时必看)
  @author iflytek
"""
import requests
import time
import hashlib
import base64
import json
import requests
import numpy as np
import base64
from naoqi import ALProxy
import vision_definitions
import time
import json
import cv2
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

ip = "127.0.0.1"
port = 9559
remote_url = "http://192.168.1.119:5678/text_recognition"
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
fps = 10
avd = ALProxy("ALVideoDevice",ip,port)
mem = ALProxy("ALMemory",ip,port)
# avd.setActiveCamera(vision_definitions.kTopCamera)
# avd.startCamera(vision_definitions.kTopCamera)
# avd.openCamera(vision_definitions.kTopCamera)
# cam_handler = avd.subscribe("mycam",vision_definitions.kVGA,vision_definitions.kRGBColorSpace,fps)
print(avd.getSubscribers())
print('================')


#from urllib import parse
# 印刷文字识别 webapi 接口地址
URL = "http://webapi.xfyun.cn/v1/service/v1/ocr/general"
# 应用ID (必须为webapi类型应用，并印刷文字识别服务，参考帖子如何创建一个webapi应用：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=36481)
APPID = "5c793546"
# 接口密钥(webapi类型应用开通印刷文字识别服务后，控制台--我的应用---印刷文字识别---服务的apikey)
API_KEY = "9c55fa9322bd6a5919ef9dcc58f145e0"
def getHeader():
#  当前时间戳
    curTime = str(int(time.time()))
#  支持语言类型和是否开启位置定位(默认否)
    param = {"language": "cn|en", "location": "false"}
    param = json.dumps(param)
    paramBase64 = base64.b64encode(param.encode('utf-8'))

    m2 = hashlib.md5()
    str1 = API_KEY + curTime + paramBase64
    m2.update(str1.encode('utf-8'))
    checkSum = m2.hexdigest()
# 组装http请求头
    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
    }
    return header
# 上传文件并进行base64位编码
# with open(r'E://1.jpg', 'rb') as f:
#     f1 = f.read()

# f1_base64 = str(base64.b64encode(f1), 'utf-8')

while True:
    camdata = avd.getImageRemote("FrameGetter/Camera_1")
    if camdata:
        img = camdata[6]
        img = [ord(v) for v in img]
        print(img[0:10])
        img = np.array(img,dtype=np.uint8)
        img = img.reshape((480,640,3))
        
        img_bin = cv2.imencode(".jpg",img)[1]
        f1_base64 = str(base64.b64encode(img_bin)).encode("utf-8")
        data = {
                'image': f1_base64
                }
        res = requests.post(URL, data=data, headers=getHeader())
        result = str(res.content).encode('utf-8')
        print(result)
    time.sleep(1/fps)



