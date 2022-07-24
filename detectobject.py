import numpy as np
import cv2
import os
def detect_object(frame):
# read the model
    cfg_path=os.path.abspath('yolo/yolov4.cfg')
    weights_path=os.path.abspath('yolov4.weights')
    names_path=os.path.abspath('yolo/coco.names')

    net=cv2.dnn_DetectionModel(cfg_path,weights_path)

    net.setInputSize(704,704)
    net.setInputScale(1.0/255)
    net.setInputSwapRB(True)
    frame=cv2.resize(frame,dsize=(704,704),interpolation=cv2.INTER_AREA)

    with open(names_path,'rt') as f:
        names=f.read().rstrip('\n').split('\n')


    classes,confidence,boxes=net.detect(frame,confThreshold=0.10,nmsThreshold=0.4)


    for classId,confidence,box in zip(classes.flatten(),confidence.flatten(),boxes):
        label='%.2f'% confidence
        label='%s: %s'%(names[classId],label)
        labelSize,baseLine=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        left,top,width,height=box
        top=max(top,labelSize[1])
        cv2.rectangle(frame,box,color=(0,0,255),thickness=5)

        cv2.putText(frame,label,(left,top),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),thickness=3)
        
    return frame

