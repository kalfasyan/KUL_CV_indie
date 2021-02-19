import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

DATADIR = './data'
cap = cv2.VideoCapture('mymovie2.mp4')
movielen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
minutes = int(duration/60)
seconds = duration%60
sec = fps # using sec for readability

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
othercorner = (10,520)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

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
    height, width, layers = image.shape
    new_h = int(round(height / 2))
    new_w = int(round(width / 2))
    resized = cv2.resize(image, (new_w, new_h))
    return resized

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

for fi in tqdm(range(movielen), desc="Recording the movie.."):
    ret, frame = cap.read()
    # frame = rotate_image(frame)
    frame = resize_image(frame)

    # Switch the movie between color and grayscale a few times (~4s)
    if sec < fi <= 2*sec or 3*sec < fi <= 4*sec:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.putText(frame, 'Grayscale', bottomLeftCornerOfText, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
    elif 4*sec < fi <= 6*sec:
        frame = cv2.GaussianBlur(frame,(5,5),0) 
        frame = cv2.putText(frame, 'GaussianBlur - kernel=(5,5)', bottomLeftCornerOfText, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
    elif 6*sec < fi <= 8*sec:
        frame = cv2.GaussianBlur(frame,(13,13),0) 
        frame = cv2.putText(frame, 'GaussianBlur - kernel=(13,13)', bottomLeftCornerOfText, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
    elif 8*sec < fi <= 10*sec:
        frame = cv2.bilateralFilter(frame, 15, 5, 5)
        frame = cv2.putText(frame, 'Bilateral filter - sigmaColor=5, sigmaSpace=5', bottomLeftCornerOfText, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
        frame = cv2.putText(frame, 'Notice how Bilateral filtering doesn\'t affect the tiles (and their edges)', othercorner, font, fontScale/2, (250,250,250), 1, cv2.LINE_AA) 
    elif 10*sec < fi <= 12*sec:
        frame = cv2.bilateralFilter(frame, 15, 75, 75)
        frame = cv2.putText(frame, 'Bilateral filter - sigmaColor=75, sigmaSpace=75', bottomLeftCornerOfText, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
        frame = cv2.putText(frame, 'Even with larger sigmas. Although, you see that the texture \"noise\" of the animal figures is gone now.', othercorner, font, fontScale/2, (250,250,250), 1, cv2.LINE_AA) 
    else:
        pass        # frame = rotate_image(frame)


    cv2.imshow('Frame', frame)
    # save_frame(frame, name=f"{DATADIR}/frame_{fi}.jpg")

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    # if fi > 2500:
    #     break
# cap.release()
cv2.destroyAllWindows()

# process_movie(movielen)





cap.release()