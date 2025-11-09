import os, glob, argparse, cv2, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import LK_Work.LucasKanade as LK
import LK_Work.InverseLucasKanadeAffine as invLK
import CNN_Work as CNN

def main():

    plt.ion()
    fig,ax = plt.subplots()

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
    CNN_FREQUENCY = 60  # runn CNN every 60 iteratiosn

    for (i, img_path) in enumerate(os.listdir(f"{base}/images_all/")):
        # print(f"{i}: {base}/images_all/{img_path}")
        print(i)
        # read in new im
        curr_img = cv2.imread(f"{base}/images_all/{img_path}")

        # if time for CNN, do CNN, else compare and update lucas kanade
        if i % CNN_FREQUENCY == 0:
            curr_bounding = CNN.lb(curr_img)
            print(f"CNN box: {curr_bounding}")
            if (curr_bounding is None):
                continue
        else:
            # curr_bounding = tuple(map(int, LK.LucasKanadeTracker(prev_img, curr_img, prev_bounding)))
            curr_bounding = tuple(map(int, invLK.InverseCompositionAffine(prev_img, curr_img, prev_bounding)))
            print(f"LK box: {curr_bounding}")
        disp_img = CNN.draw_boxes(curr_img, curr_bounding)

        # assign curr to prev
        prev_img = curr_img
        prev_bounding = curr_bounding
        
        ax.imshow(disp_img[:,:,::-1])
        plt.pause(0.05)
        ax.clear()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()