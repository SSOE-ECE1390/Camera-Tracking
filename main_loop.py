import os, glob, argparse, cv2, numpy as np
from pathlib import Path
import LK_Work.LucasKanade as LK
import CNN_Work as CNN

def main():

    ''' PRE PROCESSING'''
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

    # # import cnn algo and detect (first red car_) object
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--image_dir", default="CNN_Work/redcar_data/images_all", help="folder of frames")
    # ap.add_argument("--labels_dir", default="CNN_Work/redcar_data/labels_all", help="where to write YOLO txts")
    # ap.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained YOLO model")
    # ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    # ap.add_argument("--red_thresh", type=float, default=0.10, help="fraction of red pixels to call red_car")
    # ap.add_argument("--sat_min", type=int, default=70, help="HSV S lower bound")
    # ap.add_argument("--val_min", type=int, default=50, help="HSV V lower bound")
    # args = ap.parse_args()
    # CNN.labeler(args)

    # file stream tracking
    base = "CNN_Work/redcar_data"
    idx = 0

    for (i, img_path) in enumerate(os.listdir(f"{base}/images_all/")):
        print(f"{base}/images_all/{img_path}")
        # read in new im
        curr_img = cv2.imread(f"{base}/images_all/{img_path}")
        
        # do LK tracking stuff
        # curr_bounding = LK.LucasKanadeTracker(prev_img, curr_img, prev_bounding)

        # assign old to new
        prev_img = curr_img
        # prev_bounding = curr_bounding
    print("out")

if __name__ == "__main__":
    main()