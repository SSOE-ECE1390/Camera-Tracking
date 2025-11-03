import cv2, os, glob

IN_DIR = "raw_data"
OUT_DIR = "redcar_data/images_all"
os.makedirs(OUT_DIR, exist_ok=True)
EVERY_N = 5  # extract every 5th frame

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
