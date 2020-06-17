import cv2
import numpy as np
import random
import os
from PIL import Image
import time

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

distance_thres = 50

cap = cv2.VideoCapture('data/humans.mp4')

# class_names = [c.strip() for c in open('data/coco.names').readlines()]

def dist(pt1,pt2):
    try:
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    except:
        return

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 

_,frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output.avi', fourcc, 30,(frame.shape[1], frame.shape[0]), True)


ret = True
while ret:

    ret, img = cap.read()
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id!=0:
                continue
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    persons = []
    person_centres = []
    violate = set()

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            persons.append(boxes[i])
            person_centres.append([x+w//2,y+h//2])
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)

    for i in range(len(persons)):
        for j in range(i+1,len(persons)):
            if dist(person_centres[i],person_centres[j]) <= distance_thres:
                violate.add(tuple(persons[i]))
                violate.add(tuple(persons[j]))
    
    v = 0
    for (x,y,w,h) in persons:
        if (x,y,w,h) in violate:
            color = (0,0,255)
            v+=1
        else:
            color = (0,255,0)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.circle(img,(x+w//2,y+h//2),2,(0,0,255),2)

    cv2.putText(img,'No of Violations : '+str(v),(15,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,126,255),2)
    writer.write(img)
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
