import cv2
import threading
import queue
import time

class VideoPipeLine:
    def __init__(self, video_path, max_queue_size=128):
        self.stream = cv2.VideoCapture(video_path)
        self.stopped = False
        self.Q = queue.Queue(maxsize = max_queue_size)

    def start(self):
        t = threading.Thread(target = self.update, args = ())
        t.daemon = True
        t.start()

        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            
            if not self.Q.full():
                grabbed, frame = self.stream.read()

                if not grabbed:
                    self.stop()
                    return

                self.Q.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.Q.get()
    
    def more(self):
        return self.Q.qsize() > 0
    
    def stop(self):
        self.stopped = True
        self.stream.release()