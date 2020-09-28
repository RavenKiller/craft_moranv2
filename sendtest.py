import requests
import numpy as np
import base64
from naoqi import ALProxy
import vision_definitions

ip = "127.0.0.1"
port = 9559
avd = ALProxy("ALVideoDevice",ip,port)
cam_handler = avd.subscribeCamera(vision_definitions.kTopCamera,vision_definitions.kVGA,vision_definitions.kRGBColorSpace,10)

img = avd.getImageLocal(cam_handler)
print(img)


# print(res)