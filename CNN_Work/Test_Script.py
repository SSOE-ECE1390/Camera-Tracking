from boxVisual import draw_boxes
from labeling import main as lb
import os, glob, argparse, cv2, numpy as np
from pathlib import Path
#import LK_Work.LucasKanade as LK

def main():
    image_path = os.path.abspath("CNN_Work/Test_Images/IMG_1015_000001.jpg")
    image = cv2.imread(image_path)
    if image is None:
        print("Could not find the images")
    

    box_coords = lb(image)
    if box_coords is None:
        print("No Box cords")
    

    labled_img = draw_boxes(image, box_coords)
    if labled_img is not None:
        test_path = os.path.abspath("CNN_Work/Test_Images/Test_out.jpg")
        cv2.imwrite(test_path, labled_img)
    else:
        print("The Labled image is none")


if __name__ == "__main__":
    main()