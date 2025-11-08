import cv2
from pathlib import Path
import LK_Work.LucasKanade as LK
import CNN_Work as CNN

def main():

    # load data
    IMG_DIR = Path(r"C:/Users/novik//Documents/pitt/2025_fall/ECE_1390/Camera-Tracking/CNN Work/redcar_data/images_all")
    VID_DIR = Path(r"C:/Users/novik//Documents/pitt/2025_fall/ECE_1390/Camera-Tracking/CNN Work/raw_data/IMG_1015.MOV")

    # import cnn algo

    # detect object

    # draw initial box

    # extract feature points

    while(1):  # run while video frames exist
        print("here")
        # detect object

        # draw box around object

        # extract feature points

        # call lucas-kanade to update bounding box position


if __name__ == "__main__":
    main()