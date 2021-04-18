
import numpy as np
from scipy.optimize import fmin_cg
import math
import pickle
import cv2
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def Canny(oimg):
  img = oimg
  edges = cv2.Canny(img,60,180)
  return edges
# function to map value between 0 to 1
def sigmoid(z):
  return 1.0/(1.0 + np.exp(-z))

# hypothesis function (predicted value)
def hypo(X,theta):
  return sigmoid(X.dot(theta))

# find
def Cost(theta,X,y,lamda):
  m = X.shape[0]
  y1 = y.dot(np.log(hypo(X,theta)))
  y0 = (1-y).dot(np.log(1-hypo(X,theta)))
  temp = theta
  temp[0] = 0
  reg_term = (lamda/(2 * m)) *  (temp.dot(temp))
  J = (-1/m) * (y1+y0) + reg_term
  return J

def Fprime(theta,X,y,lamda):
  m = X.shape[0]
  ht = hypo(X,theta)
  reg_term = (lamda/m) * theta
  reg_term[0] = 0
  grad = ((ht-y).dot(X)/m) + reg_term
  return grad

def OneVsAll(X,y,lamda, numOfLabels):
  m,n = X.shape
  allTheta = np.zeros((numOfLabels,n+1))
  X = np.c_[np.ones((m,1)), X]
  for i in range(0,numOfLabels):
    initTheta = np.zeros((n+1,1))
    theta = fmin_cg(f=Cost, x0=initTheta, fprime=Fprime, args=(X, y==i, lamda), maxiter=50)
    allTheta[i,:] = theta
  return allTheta

def predict_one_vs_all(all_theta, X):
    m = len(X)
    print("Xpred",X.shape)
    X = np.hstack((np.ones((m, 1)), X))
    print("Xhpred",X.shape)
    return np.argmax(hypo(X, all_theta.T), axis=1)

# data = np.loadtxt("VaibhavParDataset.csv",delimiter=',')
# data = np.loadtxt("sanjayPartialDataset.csv",delimiter=',')
data = np.loadtxt("image6.csv",delimiter=',')

dm,dn = data.shape

seln = np.random.permutation(dm)
seln = seln.tolist()

trainSize = (int)(dm * 0.85)

kernel = np.ones((5,5),np.uint8)

X = [data[x,0:-1] for x in seln[:trainSize]]
y = [data[x,-1] for x in seln[:trainSize]]
Xt = [data[x,0:-1] for x in seln[trainSize:]]
yt = [data[x,-1] for x in seln[trainSize:]]

# X = data[:,0:-1]
# y = data[:,-1]


# over = SMOTE('minority')
# under = RandomUnderSampler('majority')
# steps = [('o', over), ('u', under)]
# pipeline = Pipeline(steps=steps)

# Xs,ys = pipeline.fit_resample(X,y)
# Xs = np.array(Xs)
# for x in X:
#   # x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
#   x=x.reshape((20,20))
#   x = Canny(x)
#   x = cv2.dilate(x,kernel,iterations = 2)
#   x = x/255

# for x in Xt:
#   # x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
#   x=x.reshape((20,20))
#   x = Canny(x)
#   x = cv2.dilate(x,kernel,iterations = 2)
#   x = x/255
#   x = x.reshape((1,400))

# Xs=np.array(Xs)

X=np.array(X)

Xt=np.array(Xt)

print(X.shape)

# ys=np.array(ys)

y=np.array(y)

print(y.shape)

yt=np.array(yt)

# print("asdfasfasfd",Xs.shape,ys.shape)
# x = X
# plt.hist(x.reshape(-1))
# plt.show()
lamda = 1.13
numOfLabels = 10

allT = OneVsAll(X,y,lamda,numOfLabels)

# frame = Xt[0:20,:]
# yi = yt[0:20]
# img = frame.reshape((20,20))
# img = cv2.resize(img,(200,200),interpolation=cv2.INTER_AREA)

# pickimg = open("imgp.pickle","wb")
# pickle.dump(frame,pickimg)
# pickimg.close()
# i=0
# for ex in frame:

#   ex = np.r_[1,ex]
#   pred = hypo(ex,allT.T)
#   res = np.argmax(pred,axis=0)
#   print("predicted-->",res)
#   print("actual-->",yt[i])
#   if(input("next")=='q'):
#     break
#   else:
#     pass
#   i+=1
# img = Xs[0,:]
# # print(img)
# img = img.reshape((20,20))
# img = cv2.resize(img,(200,200))
# # img = np.array(img)
# cv2.imshow("fasdf",img)

# cv2.waitKey(5000)

print("allt-->",allT.shape)
pred = predict_one_vs_all(allT, Xt)
print("pred-->",pred.shape)
# fig, ax = plt.subplots()
# plt.plot(pred,ys,'rx')
# plt.axis([0,10,0,10])
# ax.set_title("asdf")
# ax.legend()
# plt.show()
print('Training Set Accuracy: {}'.format(np.mean((pred == yt)) * 100))

pickle_in  = open("weightsmix.pickle","wb")
pickle.dump(allT,pickle_in)
pickle_in.close()