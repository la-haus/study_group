"""Periodically capture images from camera

Prerequisites:
  - sudo apt-get install libgtk2.0-dev
  - conda install -c conda-forge opencv=4.1.0
  - pip install opencv-contrib-python  : Needed to get cv2.data submodule
"""
import os
import cv2
import time
import datetime as dt
from pathlib import Path

import numpy as np
from cv2 import VideoCapture
import pygame

# type alias
Image = np.ndarray  # A CV2-images is really just an array

CAMERA_IDX = 2  # Necessary when there is more than one webcam in the system, otherwise just use 0
SLOUCH_THRESHOLD = 0.15
ALERT_SOUND_FILE = os.getenv('HOME') + '/suspend-error.oga'
FACE_DETECT_MODEL_SPEC = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

DATA_PATH = Path(os.getenv('HOME')) / '_data/bad-posture-detect'
DATA_PATH.mkdir(parents=True, exist_ok=True)

STORE_IMGS = False
# %%


def main():
    """run the main loop"""
    # %%
    pygame.mixer.init()
    cam = VideoCapture( CAMERA_IDX )  # get camera handle

    detector = SlouchDetector( SLOUCH_THRESHOLD, do_store_imgs=STORE_IMGS )

    try:
        while True:
            img = _capture_img(cam)
            # an (color) image is just a Width x Height x NumChannels 'matrix',
            # really a rank-3 tensor
            # channels are Blue, Green, Red (CV2 ideasyncratic ordering...)
            # print( type(img),  img.shape )   # =>  <ndarray>  (480, 640, 3)
            # Convert into grayscale
            # an grayscale image is just a Width x Height x NumChanels matrix,
            # really a rank-2 tensor
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(type(gray), gray.shape)   # =>  <ndarray>  (480, 640)
            detector.detect( gray )
            time.sleep(1)

    except Exception as exc:
        cam.release()  # close the webcam
        raise exc


class SlouchDetector:
    """Object used to detect a face in a grayscale image,  measure its vertical position
    and determine whether there is slouching"""

    def __init__(self, thresh: float, do_store_imgs: bool = False ):
        self.reference_y = None
        self.thresh = thresh
        self.face_cascade = cv2.CascadeClassifier(FACE_DETECT_MODEL_SPEC)
        self.do_store_imgs = do_store_imgs

    def detect(self, gray: Image):
        """Detect the main face in the image and whether it is slouching
        as compared to first detection"""
        now = dt.datetime.now()
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            face = faces[0]
            face_height = face[3]
            face_y = (face[1] + face_height * 0.5)

            if self.reference_y is None:
                self.reference_y = face_y
                print(f'{now} reference_y: {self.reference_y} thresh: {self.thresh}')

            ratio = -(face_y - self.reference_y) / face_height
            is_slouching = ratio < -self.thresh

            if is_slouching:
                print(f'{now} y:{face_y} h:{face_height} ratio:{ratio:.4f} you are slouching!!!')
                play_alert_sound()
            else:
                print(f'{now} y:{face_y}, h:{face_height} ratio:{ratio:.4f} you are OK')

            if self.do_store_imgs:
                self._draw_face_frame(face, gray, is_slouching)

        else:
            print(now, 'no faces detected')

    def _draw_face_frame(self, face, gray: Image, is_slouching: bool):
        """Draw rectangles around the faces"""
        x, y, w, h = face

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if is_slouching:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(rgb, (x, y), (x + w, y + h), color, 2)

        line_x = int(x + w + 10)
        face_y = int(y + h / 2)
        cv2.line(rgb, (line_x, 0), (line_x, face_y), color, 2)
        cv2.line(rgb, (line_x - 5, face_y), (line_x + 5, face_y), color, 2)
        cv2.putText(rgb, f"face_y = {face_y:.1f}", (line_x + 15, int((y + h / 2) / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 1.4, color)

        face_height = face[3]
        face_y2 = (face[1] + face_height * 0.5)
        ratio = -(face_y2 - self.reference_y) / face_height
        is_slouching2 = ratio < -self.thresh

        diff = face_y2 - self.reference_y
        labels = [ f"face_y = {face_y2:.0f} ref_y:{self.reference_y:.1f} diff:{diff}",
                   f"f_height={face_height:d} ratio={ratio:.4f} is_slouching2={is_slouching2}"]
        cv2.putText(rgb, labels[0], (10, gray.shape[0] - 22), cv2.FONT_HERSHEY_PLAIN, 1.4, color)
        cv2.putText(rgb, labels[1], (10, gray.shape[0] - 5 ), cv2.FONT_HERSHEY_PLAIN, 1.4, color)

        now_str = dt.datetime.now().strftime("%H-%M-%S")
        cv2.imwrite(str(DATA_PATH / f'img_{now_str}_{int(is_slouching)}.jpg'), rgb)


def _interactive_testing():
    # %%
    # noinspection PyUnresolvedReferences
    runfile('bad-posture-detect/detector.py')
    # %%
    face_cascade = cv2.CascadeClassifier(FACE_DETECT_MODEL_SPEC)
    # Load the cascade detector (this is classical computer vision, not neural-network based!)
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # %%
    # Read the input image
    cam = VideoCapture( CAMERA_IDX )
    # %% GUI sound production
    pygame.mixer.init()

    # %%
    img = _capture_img(cam)
    # an (color) image is just a Width x Height x NumChannels 'matrix', really a rank-3 tensor
    # channels are Blue, Green, Red (CV2 ideasyncratic ordering...)
    print( type(img),  img.shape )   # =>  <ndarray>  (480, 640, 3)
    # _interactive_show_img( img )
    # Convert into grayscale
    # an grayscale image is just a Width x Height x NumChanels matrix, really a rank-2 tensor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(type(gray), gray.shape)   # =>  <ndarray>  (480, 640)
    # %%
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print( faces )

    _draw_face_frames( faces, img )
    # %%
    cv2.imwrite( str(DATA_PATH / "img_and_faces.jpg"), img )
    # %%
    cam.release()  # close the webcam
    # %%
    play_alert_sound()
    # %%


def _draw_face_frames( faces, img ):
    """Draw rectangles around the faces"""
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # %%


def _capture_img( cam: cv2.VideoCapture ):
    s, img = cam.read()
    if not s:
        raise RuntimeError('Failed to get img from camera')
    return img
    # %%


def _interactive_show_img( img, window_title='cam-img' ):
    cv2.imshow(window_title, img)
    cv2.waitKey()
    # %%


def play_alert_sound():
    """play the sound pointed to by ALERT_SOUND_FILE"""
    # %%
    sound = pygame.mixer.Sound( ALERT_SOUND_FILE )
    sound.play()
    # %%


# %%

main()
