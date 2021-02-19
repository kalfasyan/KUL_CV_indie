import cv2
import numpy as np
import glob
import os
from tqdm import tqdm


print(f"\n####################################\n\nWelcome to the greatest movie of all time..\nswitch off your phones and enjoy!\n\n####################################\n")
print(f"Made by Ioannis Kalfas (ioannis.kalfas@kuleuven.be)")
print(f"using OpenCV version: {cv2.__version__}")

DATADIR = './data'
cap = cv2.VideoCapture('mymovie.mp4')
movielen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

print(f'Movie details:\n')
print(f'FPS = {fps:.1f}')
print(f'# of frames = {frame_count}')
print(f'Duration (secs) = {duration:.0f}')
minutes = int(duration/60)
seconds = duration%60
print(f'Duration (mins, secs) = {minutes:.0f} minutes {seconds:.0f} seconds')

def save_all_frames(length=0):
    cf = 0
    while cf < length:
        ret, frame = cap.read()
        name = f"{DATADIR}/frame_{cf}.jpg"
        print(f"Saving frame {name}")
        cv2.imwrite(name, frame)
        cf += 1

    cap.release()
    cv2.destroyAllWindows()

# save_all_frames(movielen)





cap.release()