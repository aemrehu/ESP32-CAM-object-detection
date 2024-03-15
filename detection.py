import cv2
import numpy as np
import urllib.request

print("Start")

url = 'http://192.168.10.42:81'

cap = cv2.VideoCapture(url)
whT=320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile='coco.names'
classNames=[]
with open(classesfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

print("File opened")

modelConfig = 'yolov3.cfg'
modelWeights= 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("Model loaded")

def findObject(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    # print(indices)

    for i in indices:
        # i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)


print("Start loop")

while True:
    sucess, img= cap.read()
    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)
    layernames=net.getLayerNames()
    # print(layernames)
    outputNames = [layernames[i-1] for i in net.getUnconnectedOutLayers()]

    #print(net.getUnconnectedOutLayers())
    outputs = net.forward(outputNames)
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    #print(outputs[0][0])
    findObject(outputs,img)



    cv2.imshow('IMage',img)
    cv2.waitKey(1)

# while True:
#     img_resp=urllib.request.urlopen(url)
#     print("Image response")
#     imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
#     im = cv2.imdecode(imgnp,-1)
#     sucess, img= cap.read()
#     blob=cv2.dnn.blobFromImage(im,1/255,(whT,whT),[0,0,0],1,crop=False)
#     net.setInput(blob)
#     layernames=net.getLayerNames()
#     print(layernames)
#     outputNames = [layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]

#     #print(net.getUnconnectedOutLayers())
#     outputs = net.forward(outputNames)
#     #print(outputs[0].shape)
#     #print(outputs[1].shape)
#     #print(outputs[2].shape)
#     #print(outputs[0][0])
#     findObject(outputs,im)



#     cv2.imshow('IMage',im)
#     cv2.waitKey(1)
