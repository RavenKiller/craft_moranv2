import tornado.ioloop
import tornado.web
import base64

class TextRecognitionHanlder(tornado.web.RequestHandler):
    def post(self):
        print(self.get_argument('image'))

def make_app():
    return tornado.web.Application([
        (r"/test_recognition", TextRecognitionHanlder),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(5678)
    tornado.ioloop.IOLoop.current().start()