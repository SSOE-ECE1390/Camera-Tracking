import os, glob, argparse, cv2, numpy as np
from pathlib import Path
import LK_Work.LucasKanade as LK
import CNN_Work as CNN

def main():

    ''' FRAME EXTRACTING '''
    # # extract frames from video to CNN_Work/redcar_data/images_all and labels to CNN_Work/redcar_data/labels_all
    # IN_DIR = "CNN_Work/raw_data"
    # OUT_DIR = "CNN_Work/redcar_data/images_all"
    # os.makedirs(OUT_DIR, exist_ok=True)
    # EVERY_N = 5

    # for vp in glob.glob(os.path.join(IN_DIR, "*.MOV")):
    #     cap = cv2.VideoCapture(vp)
    #     base = os.path.splitext(os.path.basename(vp))[0]
    #     i = kept = 0
    #     while True:
    #         ok, frame = cap.read()
    #         if not ok: break
    #         if i % EVERY_N == 0:
    #             fn = f"{base}_{kept:06d}.jpg"
    #             cv2.imwrite(os.path.join(OUT_DIR, fn), frame)
    #             kept += 1
    #         i += 1
    #     cap.release()
    # print("done")

    ''' MAIN LOOP '''
    base = "CNN_Work/redcar_data"
    idx = 0

    for (i, img_path) in enumerate(os.listdir(f"{base}/images_all/")):
        print(f"{i}: {base}/images_all/{img_path}")
        # read in new im
        curr_img = cv2.imread(f"{base}/images_all/{img_path}")
        
        # do LK tracking stuff
        # curr_bounding = LK.LucasKanadeTracker(prev_img, curr_img, prev_bounding)

        # assign old to new
        prev_img = curr_img
        # prev_bounding = curr_bounding

if __name__ == "__main__":
    main()