
import numpy as np
from scipy.optimize import fmin_cg
import math
import pickle
import cv2
import matplotlib.pyplot as plt


J_hist = [[0 for j in range(1)] for i in range(10)]# J_hist = np.array(J_hist)
print(J_hist)
print(len(J_hist))
print(len(J_hist[0]))
# ============================= settings ===========================

# kernel = np.ones((5,5),np.uint8)
lamda = 1
numOfLabels = 10


# ====================== pre-processing funtions=================


# def Canny(oimg):
#   img = oimg
#   edges = cv2.Canny(img,60,180)
#   return edges
# ==================================================================

#======================= helper functions ======================

# function to convert raw values to readable formats like probability
def sigmoid(z):
  return 1.0/(1.0 + np.exp(-z))


# hypothesis function (predicted value)
def hypo(X,theta):
  return sigmoid(X.dot(theta))


# cost function

def Cost(theta,X,y,lamda,i):
  m = X.shape[0]
  y1 = y.dot(np.log(hypo(X,theta)))
  y0 = (1-y).dot(np.log(1-hypo(X,theta)))
  temp = theta
  temp[0] = 0
  reg_term = (lamda/(2 * m)) *  (temp.dot(temp))
  J = (-1/m) * (y1+y0) + reg_term
  print("cost-->",J)
  global J_hist
  # print(J_hist.shape)
  # J_hist = np.append(J_hist[i],[J])
  J_hist[i].append(J)
  return J


# gradient(1st derivative of loss function)

def Fprime(theta,X,y,lamda,i):
  m = X.shape[0]
  ht = hypo(X,theta)
  reg_term = (lamda/m) * theta
  reg_term[0] = 0
  grad = ((ht-y).dot(X)/m) + reg_term
  return grad
# ===========================================================


# Classification algorithm

def OneVsAll(X,y,lamda, numOfLabels):
  m,n = X.shape
  allTheta = np.zeros((numOfLabels,n+1))
  X = np.c_[np.ones((m,1)), X]
  for i in range(0,numOfLabels):
    initTheta = np.zeros((n+1,1))
    theta = fmin_cg(f=Cost, x0=initTheta, fprime=Fprime, args=(X, y==i, lamda,i), maxiter=50)
    allTheta[i,:] = theta
  return allTheta


# predicts and return the most likely number
def predict_one_vs_all(all_theta, X):
    m = len(X)
    print("Xpred",X.shape)
    X = np.hstack((np.ones((m, 1)), X))
    print("Xhpred",X.shape)
    return np.argmax(hypo(X, all_theta.T), axis=1)



data = np.loadtxt("image6.csv",delimiter=',')

dm,dn = data.shape

seln = np.random.permutation(dm)
seln = seln.tolist()

trainSize = (int)(dm * 0.75)


X = [data[x,0:-1] for x in seln[:trainSize]]
y = [data[x,-1] for x in seln[:trainSize]]
Xt = [data[x,0:-1] for x in seln[trainSize:]]
yt = [data[x,-1] for x in seln[trainSize:]]




X=np.array(X)

Xt=np.array(Xt)

print(X.shape)


y=np.array(y)

print(y.shape)

yt=np.array(yt)


allT = OneVsAll(X,y,lamda,numOfLabels)


pred = predict_one_vs_all(allT, Xt)

print('Training Set Accuracy: {}'.format(np.mean((pred == yt)) * 100))

names = ["class 0","class 1","class 2","class 3","class 4","class 5","class 6","class 7","class 8","class 9"]
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
for i in range(1,11):
  plt.subplot(2,5,i)
  plt.xlabel("iterations",fontdict=font2)
  plt.ylabel("cost",fontdict=font2)
  plt.plot(J_hist[i-1])
  plt.title(names[i-1],fontdict=font1)
plt.show()
# pickle_in  = open("weightsfinal.pickle","wb")
# pickle.dump(allT,pickle_in)
# pickle_in.close()