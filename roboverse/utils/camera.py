import os
from threading import Thread
import time
import cv2
import numpy as np

class Camera(object):
    def __init__(self, PATH, src=0, width=1920, height=1080, fps=30):
        self.capture = cv2.VideoCapture(src, apiPreference=cv2.CAP_AVFOUNDATION)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
       
        # Set FPS wait time
        self.fps = fps
        self.SECONDS_PER_IMAGE = 1/fps
        self.WAIT_MS = int(self.SECONDS_PER_IMAGE * 1000)
        
        # Initialize VideoWriter
        self.video_writer = None
        self.is_recording = False
        self.output_filename = f'{PATH}/output.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Start frame retrieval thread
        self.thread_active = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        # Set Intrinsics parameters
        self.mtx = None
        self.dist = None

        self.isNewFrame = False

    def update(self):
        """
        Update current frame  
        """
        while self.thread_active:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                if self.is_recording:
                    self.video_writer.write(self.frame)
                self.isNewFrame = True
            time.sleep(self.SECONDS_PER_IMAGE)
    
    def get_frame(self):
        """
        Returns the current frame
        """
        if self.isNewFrame:
            self.isNewFrame = False
            return self.frame
        return None

    def show_frame(self):
        """
        Shows the current frame
        """
        cv2.imshow('frame', self.frame)

    def remove_dist(self, frame):
        """
        Removes distortion from a frame

        Parameters:
        ----------
        frame : np.array
            The frame to remove distortion from.
        
        Returns:
        -------
        dst : np.array
            The frame without distortion.
        """
        if self.mtx is None or self.dist is None:
            return frame
        else:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            dst = cv2.undistort(frame, self.mtx, self.dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            return dst
        
    def start_recording(self, filename=None):
        """
        Start recording the video
        """
        if filename:
            self.output_filename = filename
        self.video_writer = cv2.VideoWriter(self.output_filename, self.fourcc, self.fps,
                                            (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                             int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        self.is_recording = True

    def stop_recording(self):
        """
        Stop recording the video
        """
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def stop_cam(self):
        """
        Stops the camera
        """
        self.stop_recording()
        self.thread_active = False
        self.capture.release()
        cv2.destroyAllWindows()