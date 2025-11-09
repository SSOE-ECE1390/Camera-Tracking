import cv2
from pathlib import Path
import glob





def draw_boxes(img, box_coords):
    h, w = img.shape[:2]
    if box_coords is None:
        return None
    
    #Top left = (x1, y1), bottom right = (x2, y2)
    x1, y1, x2, y2 = box_coords

    box_color = (0,0,225) #Red
    #draw box on image
    cv2.rectangle(img, (x1,y1), (x2,y2), box_color, 2)
    label = "red_car"

    cv2.putText(img, label, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    return img


