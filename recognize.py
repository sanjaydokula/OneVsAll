import cv2
import numpy as np
import pickle
import math
from matplotlib import pyplot as plt
import time

def sigmoid(z):
  return 1.0/(1.0 + np.exp(-z))

def hypo(X,theta):
    return sigmoid(X.dot(theta))

def Canny(oimg):
  img = oimg
  edges = cv2.Canny(img,60,180)
  return edges


pickle_out = open("weightsmix.pickle","rb")
allT = pickle.load(pickle_out)
pickle_out.close()


cap = cv2.VideoCapture(0)
dim = (20,20)

kernel = np.ones((5,5),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

while(cap.read()):
    sucs, frame = cap.read()

    gframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cframe = Canny(gframe)

    dframe = cv2.dilate(cframe,kernel,iterations = 1)
    dframe = dframe/255

    fframe = dframe
    dframe = cv2.resize(dframe, dim, interpolation=cv2.INTER_AREA)
    dframe = np.array(dframe)
    dframe = dframe.reshape((1,400))
    dframe = np.hstack((np.ones((1, 1)), dframe))

    pred = hypo(dframe,allT.T)
    first = np.argmax(pred,axis=1)
    print(first)
    fframe = cv2.putText(fframe, str(first), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("dframe",fframe)
    time.sleep(.2)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
