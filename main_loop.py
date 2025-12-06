import os, glob, argparse, cv2, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import LK_Work.LucasKanade as LK
import CNN_Work as CNN

DO_FRAME_EXTRACING = False  # toggle this to true to run on new video feed [MUST BE SET TO TRUE ON FIRST RUN WITH NEW DATA]
CNN_FREQUENCY = 60  # sets the frame interval for CNN runs bewteen LK

'''
MAIN LOOP FUNCTION: Combines Lucas-Kanade and CNN algorithms to implement tracking framework.
                    If doFrameExtracing parameter is set to True, the frame extracing procedure is done prior to the tracking.
INPUTS:
doFrameExtracing: boolean, toggles FRAME EXTRACING procedure to pre-process new video frames
OUTPUTS:
None, but displays images as algorithm computes them      
'''
def main(doFrameExtracting):

    plt.ion()
    fig,ax = plt.subplots()

    ''' FRAME EXTRACTING '''
    if doFrameExtracting:
        # extract frames from video to CNN_Work/redcar_data/images_all and labels to CNN_Work/redcar_data/labels_all
        IN_DIR = "CNN_Work/raw_data"
        OUT_DIR = "CNN_Work/redcar_data/images_all"
        os.makedirs(OUT_DIR, exist_ok=True)
        EVERY_N = 5
    
        for vp in glob.glob(os.path.join(IN_DIR, "*.MOV")):
            cap = cv2.VideoCapture(vp)
            base = os.path.splitext(os.path.basename(vp))[0]
            i = kept = 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                if i % EVERY_N == 0:
                    fn = f"{base}_{kept:06d}.jpg"
                    cv2.imwrite(os.path.join(OUT_DIR, fn), frame)
                    kept += 1
                i += 1
            cap.release()
        print("done")

    ''' MAIN LOOP '''
    base = "CNN_Work/redcar_data"
    idx = 0
    

    for (i, img_path) in enumerate(os.listdir(f"{base}/images_all/")):
        print(f"{i}: {base}/images_all/{img_path}")
        # read in new im
        curr_img = cv2.imread(f"{base}/images_all/{img_path}")

        # if time for CNN, do CNN, else compare and update lucas kanade
        if i % CNN_FREQUENCY == 0:
            curr_bounding = CNN.lb(curr_img)
            if (curr_bounding is None):
                continue
        else:
            curr_bounding = LK.LucasKanadeTracker(prev_img, curr_img, prev_bounding)
        curr_bounding = [int(v) for v in curr_bounding]
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
    doFrameExtracing = False
    main(doFrameExtracing)