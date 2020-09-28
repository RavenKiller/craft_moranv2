import requests
import numpy as np
import base64
from naoqi import ALProxy
import vision_definitions
import time
import json

ip = "127.0.0.1"
port = 9559
remote_url = "http://192.168.1.119:5678/text_recognition"
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
fps = 1
avd = ALProxy("ALVideoDevice",ip,port)
mem = ALProxy("ALMemory",ip,port)
avd.setActiveCamera(vision_definitions.kTopCamera)
avd.startCamera(vision_definitions.kTopCamera)
avd.openCamera(vision_definitions.kTopCamera)
cam_handler = avd.subscribe("mycam",vision_definitions.kVGA,vision_definitions.kRGBColorSpace,fps)
print(avd.getSubscribers())
print('================')
try:
    while True:
        img = avd.getImageRemote(cam_handler)
        print(img[0])
        if img:
            print("sending")
            print(len(img[6]))
            print(img[6][0:10])
            res = requests.post(remote_url,headers=headers, data={"image":base64.b64encode(img[6])})
            avd.releaseImage(cam_handler)
        time.sleep(1/fps)
except:
    avd.unsubscribe(cam_handler)