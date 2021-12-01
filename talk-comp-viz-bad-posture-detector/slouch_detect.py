"""Periodically capture images from camera, detect a face, check if the face has moved down
from it's original possition. Play an alert soundif it has...
"""

from typing import Tuple, List
import os
import sys
import cv2
import time
import datetime as dt
from pathlib import Path

import numpy as np
from cv2 import VideoCapture
import pygame
from pygame import Surface
from pygame import surfarray

# type aliases
Image = np.ndarray  # A CV2-images is really just an array
BBox = List[int]

CAMERA_IDX = 2  # Necessary when there is more than one webcam in the system, otherwise just use 0
SLOUCH_THRESHOLD = 0.15
ALERT_SOUND_FILE = './complete.oga'
FACE_DETECT_MODEL_SPEC = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CAM_IMG_WIDTH_HEIGHT = (640, 480)  # used for displaying the camera image in real-time

STORE_IMGS = False  # Switch to override CLI option
# %%


def main():
    """run the main loop"""
    display, cam = init_interface(camera_idx=CAMERA_IDX)
    store_imgs = ('--store-imgs' in sys.argv) or STORE_IMGS

    detector = SlouchDetector( SLOUCH_THRESHOLD, do_store_imgs=store_imgs, display=display )

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


def init_interface( camera_idx: int ) -> Tuple[Surface, VideoCapture]:
    """Initialize CV2's VideoCapture object as well as pygame's display, sound mixer, etc...
    return some of these in a tuple"""

    pygame.mixer.init()
    cam = VideoCapture( camera_idx )  # get camera handle
    cam.set( cv2.CAP_PROP_BUFFERSIZE, 1 )

    display = pygame.display.set_mode( (640, 480) )
    print( f'display={type(display)}' )
    display.fill( (255, 255, 255) )  # fill with white back-ground
    pygame.display.set_caption( 'Slouch-Detect' )

    return display, cam


class SlouchDetector:
    """Object used to detect a face in a grayscale image,  measure its vertical position
    and determine whether there is slouching"""

    def __init__(self, thresh: float, do_store_imgs: bool = False, display=None ):
        self.reference_y = None
        self.thresh = thresh
        self.face_cascade = cv2.CascadeClassifier(FACE_DETECT_MODEL_SPEC)
        self.display = display
        self.do_store_imgs = do_store_imgs

        if self.do_store_imgs:
            self.data_path = Path(os.getenv('HOME')) / '_data/bad-posture-detect'
            self.data_path.mkdir( parents=True, exist_ok=True )

            print(f'storing images under: {self.data_path}')

    def detect(self, gray: Image):
        """Detect the main face in the image and whether it is slouching
        as compared to first detection"""
        # now_str only for logging purposes
        now_str = dt.datetime.now().strftime('%M/%d %H:%M:%S.%f')[:-3]

        # Face detection happens here.
        # The following returns a (possible empty) array of bounding boxes
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        rgb_img = cv2.cvtColor( gray, cv2.COLOR_GRAY2RGB )

        if len(faces) > 0:
            face = faces[0]
            face_height = face[3]
            face_y = (face[1] + face_height * 0.5)

            if self.reference_y is None:
                self.reference_y = face_y
                print(f'{now_str} Set reference_y: {self.reference_y} thresh: {self.thresh}')

            ratio = -(face_y - self.reference_y) / face_height
            is_slouching = ratio < -self.thresh

            if is_slouching:
                conclusion = 'you are slouching'
                play_alert_sound()
            else:
                conclusion = 'you are OK!'

            print(f'{now_str} y:{face_y:4.0f} h:{face_height:4.0f} ratio:{ratio:7.4f} {conclusion}')

            rgb_img = self._draw_face_frame( face, rgb_img, is_slouching )

        else:
            print(now_str, 'no faces detected')

        self._refresh_display(rgb_img)

    def _draw_face_frame(self,  face: BBox, rgb: Image, is_slouching_: bool) -> Image:
        """Draw rectangles around the faces"""

        x, y, w, h = face

        if is_slouching_:
            color = (255, 0, 0)  # red
        else:
            color = (0, 255, 0)  # green

        cv2.rectangle(rgb, (x, y), (x + w, y + h), color, 2)

        line_x = int(x + w + 10)
        face_y = int(y + h / 2)
        cv2.line(rgb, (line_x, 0), (line_x, face_y), color, 2)
        cv2.line(rgb, (line_x - 5, face_y), (line_x + 5, face_y), color, 2)
        cv2.putText(rgb, f"face_y = {face_y:.0f}", (line_x + 15, int((y + h / 2) / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 1.4, color)

        face_height = face[3]
        face_y2 = (face[1] + face_height * 0.5)
        ratio = -(face_y2 - self.reference_y) / face_height
        is_slouching = ratio < -self.thresh

        diff = face_y2 - self.reference_y
        labels = [ f"face_y = {face_y2:.0f} ref_y:{self.reference_y:.0f} diff:{diff}",
                   f"f_height={face_height:d} ratio={ratio:.4f} slouching={is_slouching}"]
        cv2.putText(rgb, labels[0], (10, rgb.shape[0] - 22), cv2.FONT_HERSHEY_PLAIN, 1.4, color)
        cv2.putText(rgb, labels[1], (10, rgb.shape[0] - 5 ), cv2.FONT_HERSHEY_PLAIN, 1.4, color)

        if self.do_store_imgs:
            now_str = dt.datetime.now().strftime("%H-%M-%S")
            rgb2 = cv2.cvtColor( rgb, cv2.COLOR_RGB2BGR )
            cv2.imwrite(str(self.data_path / f'img_{now_str}_{int(is_slouching)}.jpg'), rgb2)

        return rgb

    def _refresh_display( self, rgb_img: Image ):
        if self.display is not None:
            rgb_img_disp = rgb_img.transpose( (1, 0, 2) )
            img_surf = surfarray.make_surface( rgb_img_disp )
            self.display.blit( img_surf, (0, 0) )

            pygame.display.update()


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
    cv2.imwrite( f"{os.getenv('HOME')}/_data/bad-posture-detect/img_and_faces.jpg", img )
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


def play_alert_sound():
    """play the sound pointed to by ALERT_SOUND_FILE"""
    sound = pygame.mixer.Sound( ALERT_SOUND_FILE )
    sound.play()


main()
