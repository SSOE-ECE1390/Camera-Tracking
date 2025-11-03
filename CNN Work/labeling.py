import os, glob, argparse, cv2, numpy as np
from ultralytics import YOLO

def red_ratio(bgr, sat_min=70, val_min=50):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,   sat_min, val_min])
    upper1 = np.array([10,  255,     255])
    lower2 = np.array([170, sat_min, val_min])
    upper2 = np.array([180, 255,     255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    return (mask > 0).mean()

def main(args):
    img_dir = args.image_dir
    out_lbl = args.labels_dir
    os.makedirs(out_lbl, exist_ok=True)

    model = YOLO(args.model)  

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                       glob.glob(os.path.join(img_dir, "*.png")) +
                       glob.glob(os.path.join(img_dir, "*.jpeg")))

    kept = 0
    for ip in img_paths:
        img = cv2.imread(ip)
        if img is None:
            print(f"[skip] cannot read {ip}")
            continue
        h, w = img.shape[:2]

        res = model.predict(img, conf=args.conf, iou=0.5, verbose=False)[0]
        if res.boxes is None:
            # write empty label file
            base = os.path.splitext(os.path.basename(ip))[0]
            open(os.path.join(out_lbl, base + ".txt"), "w").close()
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        clss  = res.boxes.cls.cpu().numpy()

        lines = []
        for (x1,y1,x2,y2), c in zip(boxes, clss):
            if int(c) != 2:
                continue

            x1i, y1i, x2i, y2i = map(int, [x1,y1,x2,y2])
            x1i = max(0, x1i); y1i = max(0, y1i)
            x2i = min(w-1, x2i); y2i = min(h-1, y2i)
            if x2i <= x1i or y2i <= y1i:
                continue

            crop = img[y1i:y2i, x1i:x2i]
            rr = red_ratio(crop, sat_min=args.sat_min, val_min=args.val_min)

            cls_id = 0 if rr >= args.red_thresh else 1  

            cx = ((x1i + x2i) / 2.0) / w
            cy = ((y1i + y2i) / 2.0) / h
            bw = (x2i - x1i) / w
            bh = (y2i - y1i) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        base = os.path.splitext(os.path.basename(ip))[0]
        with open(os.path.join(out_lbl, base + ".txt"), "w") as f:
            f.write("\n".join(lines))
        kept += 1

    print(f"done. wrote labels for {kept} images to: {out_lbl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default="redcar_data/images_all", help="folder of frames")
    ap.add_argument("--labels_dir", default="redcar_data/labels_all", help="where to write YOLO txts")
    ap.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained YOLO model")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--red_thresh", type=float, default=0.10, help="fraction of red pixels to call red_car")
    ap.add_argument("--sat_min", type=int, default=70, help="HSV S lower bound")
    ap.add_argument("--val_min", type=int, default=50, help="HSV V lower bound")
    args = ap.parse_args()
    main(args)
