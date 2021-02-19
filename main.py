import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

DATADIR = './data'
cap = cv2.VideoCapture('mymovie.mp4')
movielen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
minutes = int(duration/60)
seconds = duration%60
sec = fps # using sec for readability

""" BASIC INTRO CONSOLE MESSAGES """
print(f"\n############################################\n\nWelcome to the greatest movie of all time..\
        \nswitch off your phones and enjoy!\n\n############################################\n")
print(f"Made by Ioannis Kalfas (ioannis.kalfas@kuleuven.be)")
print(f"using OpenCV version: {cv2.__version__}")
print(f'\nMovie details:\n')
print(f'\tDuration: {minutes:.0f} minutes {seconds:.0f} second(s)')
print(f'\t{frame_count} frames')
print(f'\t{fps:.0f} FPS\n')

def resize_image(image):
    (h, w) = image.shape[:2]
    h,w = int(round(h)), int(round(w))
    image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)    
    return image

def rotate_image(image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # rotate our image by -90 degrees around the image
    M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def save_frame(frame, name=''):
    print(f"Saving frame {name}")
    cv2.imwrite(name, frame)

# def process_movie(length=0):

for fi in tqdm(range(movielen), desc="Processing movie frames.."):
    ret, frame = cap.read()
    frame = rotate_image(frame)

    frame = resize_image(frame)

    # Switch the movie between color and grayscale a few times (~4s)
    if sec < fi < 2*sec or 3*sec < fi < 4*sec:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif 4*sec < fi < 8*sec:
        frame = cv2.bilateralFilter(frame, 9, 265, 265)
    elif 8*sec < fi < 12*sec:
        frame = cv2.GaussianBlur(frame,(150,150),0) 


    cv2.imshow('Frame', frame)
    # save_frame(frame, name=f"{DATADIR}/frame_{fi}.jpg")

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    if fi > 500:
        break
# cap.release()
cv2.destroyAllWindows()

# process_movie(movielen)





cap.release()