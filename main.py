from numpy.lib.function_base import median
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--movie", help="The movie file you want to record.")
args = parser.parse_args()

DATADIR = './data'
cap = cv2.VideoCapture(f'{args.movie}.mp4')
movielen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
minutes = int(duration/60)
seconds = duration%60
sec = fps # using sec for readability

""" SETTINGS FOR SUBTITLES """
font = cv2.FONT_HERSHEY_SIMPLEX
botleft = (10,500)
lowbotleft = (10,520)
fontScale = 1
fontColor = (255,255,255)
fontColorSmall = (230,230,230)
lineType = 2

""" BASIC INTRO CONSOLE MESSAGES """
print(f"\n############################################\n\nWelcome to the greatest movie of all time..\
        \nswitch off your phones and enjoy!\n\n############################################\n")
print(f"Made by Ioannis Kalfas (ioannis.kalfas@kuleuven.be)")
print(f"using OpenCV version: {cv2.__version__}")
print(f'\nMovie details:\n')
print(f'\tDuration: {minutes:.0f} minutes {seconds:.0f} second(s)')
print(f'\t{frame_count} frames')
print(f'\t{fps:.0f} FPS\n')


""" VARIOUS HELPER FUNCTIONS """
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

def edge_detection(image, setting='horizontal', preprocess=False, ks=17):
    assert setting in ['horizontal','vertical'], "Wrong setting given for edge detection!"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if preprocess:
        gray = cv2.GaussianBlur(gray,(5,5),0)
        gray = cv2.medianBlur(gray,3) 

    if setting == 'horizontal':
        edges = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=ks)
    elif setting == 'vertical':
        edges = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=ks)
    
    # edges = cv2.convertScaleAbs(edges, alpha=255/image.max())
    print(edges.shape)
    edges = cv2.merge((edges, edges, edges)).astype(np.uint8)
    image = cv2.addWeighted(image, 1, edges, .35, 0)
    return image

# We give some time for the camera to warm-up!
import time
time.sleep(1)
background=0
ks = 3

""" MOVIE RECORDING MAIN LOOP """
for fi in tqdm(range(movielen), desc="Recording the movie.."):
    ret, frame = cap.read()
    print(frame.shape)
    # frame = rotate_image(frame)
    frame = resize_image(frame)

    # Switch the movie between color and grayscale a few times (~4s)
    if fi <= sec:
        background = np.flip(frame,axis=1)    
    # elif sec < fi <= 1.5*sec or 2.5*sec < fi <= 3*sec or 3.5*sec < fi <= 4*sec:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     frame = cv2.putText(frame, 'Grayscale', botleft, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
    # # Blurring with GaussianBlur 
    # elif 4*sec < fi <= 6*sec:
    #     frame = cv2.GaussianBlur(frame,(5,5),0) 
    #     frame = cv2.putText(frame, 'GaussianBlur - kernel=(5,5)', botleft, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
    # # Blurring with GaussianBlur larger kernel
    # elif 6*sec < fi <= 8*sec:
    #     frame = cv2.GaussianBlur(frame,(13,13),0) 
    #     frame = cv2.putText(frame, 'GaussianBlur - kernel=(13,13)', botleft, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
    # # Blurring with Bilateral filter
    # elif 8*sec < fi <= 10*sec:
    #     sigmaColor, sigmaSpace = 5, 5
    #     frame = cv2.bilateralFilter(frame, 15, 25, 25)
    #     frame = cv2.putText(frame, f'Bilateral filter - sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}', botleft, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
    #     frame = cv2.putText(frame, 'Notice how Bilateral filtering doesn\'t affect the tiles (and their edges)', lowbotleft, font, fontScale/2, fontColorSmall, 1, cv2.LINE_AA) 
    # # Blurring with Bilateral filter - larger sigma
    # elif 10*sec < fi <= 12*sec:
    #     sigmaColor, sigmaSpace = 95, 95
    #     frame = cv2.bilateralFilter(frame, 15, sigmaColor, sigmaSpace)
    #     frame = cv2.putText(frame, f'Bilateral filter - sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}', botleft, font, fontScale, fontColor, lineType, cv2.LINE_AA) 
    #     frame = cv2.putText(frame, 'Even with larger sigmas. Although, you see that the texture \"noise\" of the animal figures is gone now.', lowbotleft, font, fontScale/2, fontColorSmall, 1, cv2.LINE_AA) 
    # # Object grabbing in RGB 
    # elif 12*sec < fi:
    #     mask = cv2.inRange(frame, (10, 20, 160), (100, 110, 255))

    #     mask = mask > 0
    #     red = np.zeros_like(frame, np.uint8)
    #     red[mask] = frame[mask]
    #     frame = cv2.medianBlur(red,3)

    #     if 14*sec < fi:
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    #         frame = cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel)
    #         frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, np.ones((7,7),np.uint8))
    # Sobel edge detection - horizontal
    elif 2*sec < fi <= 3*sec:
        frame = edge_detection(frame, setting='vertical', ks=19)
    elif 3*sec < fi <= 4*sec:
        frame = edge_detection(frame, setting='horizontal', ks=19)
    elif 4*sec < fi <= 5*sec:
        frame = edge_detection(frame, setting='horizontal', ks=17)
    # Sobel edge detection - vertical        
    elif 5*sec < fi <= 6*sec:
        frame = edge_detection(frame, setting='horizontal', ks=21)
    else:
        pass        # frame = rotate_image(frame)


    cv2.imshow('Frame', frame)
    # save_frame(frame, name=f"{DATADIR}/frame_{fi}.jpg")

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()

# process_movie(movielen)





cap.release()