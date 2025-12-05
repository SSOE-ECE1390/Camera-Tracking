import os, glob, argparse, cv2, numpy as np
from ultralytics import YOLO

    # This function takes a BGR image (or crop) and measures how much of it is "red".
    # It converts the image to HSV, builds a mask for red pixels, and returns
    # the fraction of pixels that are red (a value between 0 and 1).
def red_ratio(bgr, sat_min=70, val_min=50):
    #Converting the BGR Image to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    #Red loops back around in HSV so have to find lower red and higher reds
    #Hue is the color, Saturation is how much white is added 
    # Value is how much black is added 
    lower1 = np.array([0,   sat_min, val_min])
    upper1 = np.array([10,  255,     255])
    lower2 = np.array([170, sat_min, val_min])
    upper2 = np.array([180, 255,     255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    numRed = np.count_nonzero(mask)
    total_pixels = mask.size

    return numRed / total_pixels
    # This function takes an image and runs a YOLO model on it to find cars.
    # For each detected car, it checks how "red" the car is using red_ratio().
    # It returns the bounding box (x1, y1, x2, y2) of the single reddest car
    # if its red ratio is above red_thresh; otherwise it returns None.
def main(Image, model_path="yolov8n.pt", conf=0.25, red_thresh=0.10, sat_min=70, val_min=50):

    model = YOLO(model_path)  
    #Take the first 2 entries Height and width but leave color
    h, w = Image.shape[:2]
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
        
        x1i, y1i, x2i, y2i = map(int, [x1,y1,x2,y2])
        x1i = max(0, x1i); y1i = max(0, y1i)
        x2i = min(w-1, x2i); y2i = min(h-1, y2i)
        if x2i <= x1i or y2i <= y1i:
            continue

        crop = Image[y1i:y2i, x1i:x2i]
        rr = red_ratio(crop, sat_min=sat_min, val_min=val_min)

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
                
