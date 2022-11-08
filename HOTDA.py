import numpy as np
from scipy.spatial import distance
import ot
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import datasets
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import SpectralClustering
import numpy as np
#import cv2,os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import glob
from matplotlib import pyplot
import seaborn as sns
import warnings
from sklearn.manifold import TSNE

def Source_target_processing(X,y):  #grouping source (and target) data into classes (and clusters)
    S=[]
    a=[]
    mu=[]
    yc_source=[]
    classes=np.unique(y)
    k=len(classes)
    mu=np.ones(k)/k
    for i in range(k):
        C=X[y==i]
        yc_source=yc_source+list(y[y==i])
        w=np.ones(C.shape[0])/C.shape[0]
        S.append(C)
        a.append(w)
        #mu.append(C.shape[0]/X.shape[0])
    mu=np.array(mu)
    return S,a,mu,yc_source

def Hot(S,a,mu,T,b,nu,reg1,reg2):   #hierarchical formulation of OT
    W=np.zeros((len(S),len(T)))
    for i in range(len(S)):
        for j in range(len(T)):
            M=distance.cdist(S[i],T[j], metric='sqeuclidean')
            OT=ot.sinkhorn(a[i],b[j],M,reg=reg1)
            W[i][j] = np.trace(np.dot(OT.T,M))
    hot=ot.sinkhorn(mu,nu,W,reg=reg2)
    return hot,W


def Mapping(S,T,a,b,HOT,reg3):   #mapping data of each class to the corresponding cluster
    index=np.argmax(HOT,1)
    Transported_S=[]
    for i in range(len(S)):
        M=distance.cdist(S[i],T[index[i]], metric='sqeuclidean')
        OT=ot.sinkhorn(a[i],b[index[i]],M,reg=reg3)
        Transported_Source=np.linalg.inv(np.diag(OT.dot(np.ones(T[index[i]].shape[0])))).dot(OT).dot(T[index[i]])
        Transported_S=Transported_S+Transported_Source.tolist()
    return Transported_S
