import os, glob, argparse, cv2, numpy as np
from ultralytics import YOLO

def red_ratio(bgr, sat_min=70, val_min=50):
    #Converting the BGR Image to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    #Red loops back around in HSV so have to find lower red and higher reds
    #Hue is the color, Saturation is how much white is added (Lower more white), 
    # Value is how much black is added (Lower more black).
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

    # img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
    #                    glob.glob(os.path.join(img_dir, "*.png")) +
    #                    glob.glob(os.path.join(img_dir, "*.jpeg")))

    #Had to change the file path to this because images were not being found. 
    #Also you have to be in the CNN Working Dir for it to run correctly
    img_dir = os.path.abspath(args.image_dir)  # ensures full absolute path

    img_paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"[INFO] Found {len(img_paths)} images")
    for p in img_paths:
        print("  ", p)


    kept = 0
    for ip in img_paths:
        img = cv2.imread(ip)
        if img is None:
            print(f"[skip] cannot read {ip}")
            continue
        #Take the first 2 entries Height and width but leave color
        h, w = img.shape[:2]

        #Running model Only mark a car if it 25% confident or more
        #IoU Intersection over Union allow lables to overlap by only 50%
        #The prediction returns a list of different image predictions. That is because
        #you can load multiple images at the same time each pred_list[] entry is a new
        #prediction for a new image. We only use 1 image so 0th entry only.
        res = model.predict(img, conf=args.conf, iou=0.5, verbose=False)[0]
        if res.boxes is None:
            # write empty label file
            base = os.path.splitext(os.path.basename(ip))[0]
            open(os.path.join(out_lbl, base + ".txt"), "w").close()
            continue
        
        #list of bounding boxes and class ids (Cars ID = 2)
        boxes = res.boxes.xyxy.cpu().numpy()
        clss  = res.boxes.cls.cpu().numpy()

        car_data = []
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

            #cls_id = 0 if rr >= args.red_thresh else 1  

            cx = ((x1i + x2i) / 2.0) / w
            cy = ((y1i + y2i) / 2.0) / h
            bw = (x2i - x1i) / w
            bh = (y2i - y1i) / h
            
            ##Added in
            car_data.append({'red_ratio': rr,
                             'coords': (cx, cy, bw, bh),
                             'box': (x1i, y1i, x2i, y2i)})
        #debug
        if not car_data:
            print(f"No cars detected or filltered out {ip}")

        if car_data:
            #debug
            print(f"Red ratios for {ip}: {[round(car['red_ratio'], 4) for car in car_data]}")
            #finding the largest ratio of red and limiting it so only 1 car can be labled red
            red_ratios = [car['red_ratio'] for car in car_data]
            reddest_index = red_ratios.index(max(red_ratios))

            for i, car in enumerate(car_data):
                if i == reddest_index and car['red_ratio'] >= args.red_thresh:
                    cls_id = 0 
                else:
                    cls_id = 1 #other cars
                cx,cy,bw,bh = car['coords']
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
