import cv2 as cv
import os
import tkinter as tk
from tkinter import filedialog

base = os.path.dirname(os.path.abspath(__file__))

faceProto = os.path.join(base,"opencv_face_detector.pbtxt")
faceModel = os.path.join(base,"opencv_face_detector_uint8.pb")
ageProto = os.path.join(base,"age_deploy.prototxt")
ageModel = os.path.join(base,"age_net.caffemodel")
genderProto = os.path.join(base,"gender_deploy.prototxt")
genderModel = os.path.join(base,"gender_net.caffemodel")

ageList = ['(0-2)','(4-6)','(8-12)','(15-20)',
           '(25-32)','(38-43)','(48-53)','(60-100)']
genderList = ['Male','Female']

faceNet = cv.dnn.readNet(faceModel,faceProto)
ageNet = cv.dnn.readNet(ageModel,ageProto)
genderNet = cv.dnn.readNet(genderModel,genderProto)

root = tk.Tk()
root.withdraw()
file = filedialog.askopenfilename(
    title="Select Face Image",
    filetypes=[("Image files","*.jpg *.jpeg *.png")]
)

img = cv.imread(file)

blob = cv.dnn.blobFromImage(
    img,1.0,(300,300),[104,117,123],True,False
)

faceNet.setInput(blob)
det = faceNet.forward()

for i in range(det.shape[2]):
    conf = det[0,0,i,2]

    if conf > 0.7:
        h,w = img.shape[:2]
        x1 = int(det[0,0,i,3]*w)
        y1 = int(det[0,0,i,4]*h)
        x2 = int(det[0,0,i,5]*w)
        y2 = int(det[0,0,i,6]*h)

        face = img[y1:y2,x1:x2]

        blob = cv.dnn.blobFromImage(
            face,1.0,(227,227),
            (78.426,87.769,114.896),
            swapRB=False
        )

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]

        cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv.putText(img,gender+" "+age,(x1,y1-10),
                   cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

cv.imshow("Age and Gender Detection",img)
cv.waitKey(0)
cv.destroyAllWindows()
