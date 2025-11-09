import CNN_Work.boxVisual 
import CNN_Work.labeling as lb 
import os, glob, argparse, cv2, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
#import LK_Work.LucasKanade as LK

def main():
    plt.ion()
    fig,ax = plt.subplots()

      # file stream tracking
    base = "CNN_Work/redcar_data"

    for (i, img_path) in enumerate(os.listdir(f"{base}/images_all/")):
        print(f"{base}/images_all/{img_path}")
        # read in new im
        image = cv2.imread(f"{base}/images_all/{img_path}")
        if image is None:
            print("Could not find the images")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        plt.pause(0.05)
        ax.clear()
    
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()