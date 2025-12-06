import cv2, os, glob

# frame extraction description: 
# This file take in the videos that are in the raw_data folder
# Every 5th frame of the video is extracted and saved as a jpg image


IN_DIR = "raw_data"                     # directory containing input videos
OUT_DIR = "redcar_data/images_all"      # directory to save extracted frames
os.makedirs(OUT_DIR, exist_ok=True)     # Creating the directory if it doesnt exist 
EVERY_N = 5                             # extract every 5th frame

# loop that goes through all of the frames in the video and pulls out every 5th frame
for vp in glob.glob(os.path.join(IN_DIR, "*.MOV")):
    cap = cv2.VideoCapture(vp)
    base = os.path.splitext(os.path.basename(vp))[0]
    i = kept = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % EVERY_N == 0:
            #Nameing the extracted frame with kept image count.
            fn = f"{base}_{kept:06d}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, fn), frame)
            kept += 1
        i += 1
    cap.release()
print("done")
