import cv2
from pathlib import Path
import glob

IMG_DIR = Path(r"C:/Users/nickl/Camera-Tracking/CNN Work/redcar_data/images_all")
LBL_DIR = Path(r"C:/Users/nickl/Camera-Tracking/CNN Work/redcar_data/labels_all")
OUT_DIR = Path(r"C:/Users/nickl/Camera-Tracking/CNN Work/redcar_data/preview_all")


OUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    glob.glob(str(IMG_DIR / "*.jpg")) 

)

def draw_boxes(img, label_path: Path):
    h, w = img.shape[:2]
    if not label_path.exists():
        return img
    with label_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
            color = (0, 0, 255) if int(c) == 0 else (0, 255, 0)  # red_car vs other_car
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img,
                        "red_car" if int(c) == 0 else "other_car",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

count = 0
for ip in image_paths:
    ipath = Path(ip)                     
    base = ipath.stem                    
    lp = LBL_DIR / f"{base}.txt"        
    img = cv2.imread(str(ipath))
    if img is None:
        print(f"[skip] cannot read {ipath}")
        continue
    vis = draw_boxes(img, lp)
    outp = OUT_DIR / f"{base}.jpg"       
    ok = cv2.imwrite(str(outp), vis)
    if not ok:
        print(f"[warn] failed to save {outp}")
    else:
        count += 1

print(f"Preview images saved to: {OUT_DIR}  (total: {count})")
