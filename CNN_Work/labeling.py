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

def main(Image, model_path="yolov8n.pt", conf=0.25, red_thresh=0.10, sat_min=70, val_min=50):

    model = YOLO(model_path)  
    #Take the first 2 entries Height and width but leave color
    h, w = Image.shape[:2]

    #Running model Only mark a car if it 25% confident or more
    #IoU Intersection over Union allow lables to overlap by only 50%
    #The prediction returns a list of different image predictions. That is because
    #you can load multiple images at the same time each pred_list[] entry is a new
    #prediction for a new image. We only use 1 image so 0th entry only.
    res = model.predict(Image, conf=conf, iou=0.5, verbose=False)[0]
    if res.boxes is None:
        #If the CNN finds nothing return null to the main script
        print("A Car could not be found")
        return None
    print("Something was fond")
    
    #list of bounding boxes and class ids (Cars ID = 2)
    boxes = res.boxes.xyxy.cpu().numpy()
    clss  = res.boxes.cls.cpu().numpy()

    car_data = []
    for (x1,y1,x2,y2), c in zip(boxes, clss):
        if int(c) != 2:
            continue
        else:
            print("A car was found")
        
        #Top left = (x1i, y1i), bottom right = (x2i, y2i)
        x1i, y1i, x2i, y2i = map(int, [x1,y1,x2,y2])
        x1i = max(0, x1i); y1i = max(0, y1i)
        x2i = min(w-1, x2i); y2i = min(h-1, y2i)
        if x2i <= x1i or y2i <= y1i:
            continue

        crop = Image[y1i:y2i, x1i:x2i]
        rr = red_ratio(crop, sat_min=sat_min, val_min=val_min)
        
        ##Added in
        car_data.append({'red_ratio': rr,
                         'box': (x1i, y1i, x2i, y2i)})
    #debug
    if not car_data:
        return None

    if car_data:
        #finding the largest ratio of red and limiting it so only 1 car can be labled red
        max_red_ratio = -1
        reddest_box = None

        for car in car_data:
            if car['red_ratio'] > max_red_ratio:
                #For red car brian wants output format
                max_red_ratio = car['red_ratio']
                reddest_box = car['box']

        #Return the max red box coordinates
        if max_red_ratio >= red_thresh:
            print("Redest box coords", reddest_box)
            return reddest_box
        else: 
            return None
                

    #     base = os.path.splitext(os.path.basename(ip))[0]
    #     with open(os.path.join(out_lbl, base + ".txt"), "w") as f:
    #         f.write("\n".join(lines))
    #     kept += 1

    # print(f"done. wrote labels for {kept} images to: {out_lbl}")
