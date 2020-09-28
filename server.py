import tornado.ioloop
import tornado.web
import base64
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
fig = plt.figure()
ax = fig.add_axes()
class TextRecognitionHanlder(tornado.web.RequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fig=plt.figure()
        self.ax = self.fig.add_axes([0.1,0.1,0.9,0.9])
    def post(self):
        zz = self.get_argument('image')

        print("received")
        img = np.array(list(base64.b64decode(self.get_argument('image'))),dtype=np.uint8)
        img = img.reshape((480,640,3))
        self.ax.imshow(img)
        plt.savefig("test.png")
        print(img.shape)
        print(img.dtype)


def make_app():
    return tornado.web.Application([
        (r"/text_recognition", TextRecognitionHanlder),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(5678)
    tornado.ioloop.IOLoop.current().start()