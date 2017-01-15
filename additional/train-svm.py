import os
import sys
import numpy as np
import time
import math
import scipy.misc
import argparse
import misc
from sklearn import svm
from skimage.measure import regionprops

def extract_features(X, y):
    Xf = {}
    yf = {}
    for i in range(len(X)):
        pred = misc.load_image(X[i])
        predsum = pred.sum()
        if predsum == 0: continue
        features = []
        features.append(predsum)
        prop = regionprops(pred[0].astype(int)) [0]
        # for n in range(2):
        #     features.append( prop.bbox[n+2] - prop.bbox[n])
        # features.append(prop.eccentricity)
        features.append(prop.equivalent_diameter)
        # features.append(prop.major_axis_length)
        # features.append(prop.minor_axis_length)
        # features.append(prop.perimeter)
        Xf[X[i]] = features
        lbl = misc.load_image(y[i])
        yf[y[i]] = (lbl.sum()>0).astype(int)
    return Xf, yf

def rename_dir(X, dir1, dir2):
    return [X[w].replace(dir1, dir2) for w in X]

def dice_predict(pred, tgt):
    predeq = (pred >= 0.5)
    tgteq = (tgt >= 0.5)
    den = predeq.sum() + tgteq.sum()
    if den == 0: return -1
    return -2* (predeq*tgteq).sum()/den




###################################################


version=15.300203201
showdir='show'
cv=10

vshowdir=showdir+'/v'+str(version)

X_dum, y_dum, X_all, y_all = misc.load_data(1)
X_all = rename_dir(X_all, 'train', vshowdir)
# X_features,y_features = extract_features(X_all, y_all)


# for feat in range(len(X_features[X_all[0]])):
thresholds=[]
dice=[0,0]
feat=0
for i in range(len(X_all)):
    pred=misc.load_image(X_all[i])
    if pred.sum()==0: continue
    target=misc.load_image(y_all[i])
    dice[0] = dice_predict(pred*0, target)
    dice[1] = dice_predict(pred, target)
    # thresholds.append([X_features[X_all[i]][feat], dice[0], dice[1]])
    thresholds.append([pred.sum(), dice[0], dice[1]])

# 5435.0
bestth=0
bestdice=0
thresholds = sorted(thresholds,key=lambda thresholds: thresholds[0])
num=len(thresholds)
for i in range(num):
    sum=0
    for j in range(i+1):
        sum+=thresholds[j][1]
    for j in range(i+1, num):
        sum+=thresholds[j][2]
    sum/=num
    if sum<bestdice:
        bestdice=sum
        bestth=thresholds[i][0]
    print(str(thresholds[i][0])+": "+str(sum))

print(str(bestth)+": "+str(bestdice))




#######################################################################





version=15.300203201
showdir='show'
cv=10

vshowdir=showdir+'/v'+str(version)

X_dum, y_dum, X_all, y_all = misc.load_data(1)
X_all = rename_dir(X_all, 'train', vshowdir)
X_features,y_features = extract_features(X_all, y_all)


# for w in np.linspace(1,2,11):
# for w in np.linspace(1,10,11):
w=1.5    
gamma=0.001
C=0.1
for feat in range(len(X_features[X_all[0]])):
    all_mean_dice = 0
    all_mean_orig_dice = 0
    all_mean_thr_dice = 0
    #
    val_pct=1/cv
    for fold in range(1, cv+1):
        X_train, y_train, X_val, y_val = misc.load_data(val_pct=val_pct, cv=fold)
        X_train = rename_dir(X_train, 'train', vshowdir)
        X_val = rename_dir(X_val, 'train', vshowdir)
        #
        X = []; y = []
        for i in range(len(X_train)):
            if not X_train[i] in X_features: continue
            X.append(X_features[X_train[i]])
            y.append(y_features[y_train[i]])
        #
        Xv = []; yv = [];
        for i in range(len(X_val)):
            if not X_val[i] in X_features: continue
            Xv.append(X_features[X_val[i]])
            yv.append(y_features[y_val[i]])
        #
        X=np.array([w[feat] for w in X]).reshape(-1, 1)
        Xv=np.array([w[feat] for w in Xv]).reshape(-1, 1)
        # SVM
        # clf = svm.SVC(gamma=gamma, C=C)
        clf = svm.SVC(kernel='linear', class_weight={0: w, 1: 1})
        clf.fit(X,y)
        yp = clf.predict(Xv)
        #
        orig_acc = len(np.where(yv)[0])/len(yv)
        acc = len(np.where(yp==yv)[0])/len(yv)
        #
        mean_dice = 0
        mean_orig_dice = 0
        mean_thr_dice = 0
        n = 0
        dice=[0,0]
        for i in range(len(X_val)):
            pred=misc.load_image(X_val[i])
            if pred.sum()==0: continue
            target=misc.load_image(y_val[i])
            dice[0] = dice_predict(pred*0, target)
            dice[1] = dice_predict(pred, target)
            mean_dice += dice[yp[n]]
            mean_orig_dice += dice[1]
            mean_thr_dice += dice[pred.sum()>5000]
            n+=1
        #
        mean_dice/=n
        mean_orig_dice/=n
        mean_thr_dice/=n
        all_mean_dice+=mean_dice
        all_mean_orig_dice+=mean_orig_dice
        all_mean_thr_dice+=mean_thr_dice
    #
    all_mean_dice/=cv
    all_mean_orig_dice/=cv
    all_mean_thr_dice/=cv
    print("w "+str(w)+" mean_dice: "+str(all_mean_dice)+" thr: "+str(all_mean_thr_dice)+" original: "+str(all_mean_orig_dice)+"\n")











# clf = svm.SVC(gamma=gamma, C=C)
# clf.fit(X,y)

# import pickle
# fn='params/v'+str(version)'/svm.pklz'
# with open(fn, 'wb') as wr:
#     pickle.dump(clf, wr)
#     wr.close()


# X_feat=np.array([w[0,3:8,10] for w in X]).reshape(-1, 1)




""" test
version=15.300203201
showdir='submit'
outputdir='submit-svm'
cv=10
# def train_model(version, showdir='show', cv=10):

vshowdir=showdir+'/v'+str(version)
outputdir=showdir+'/v'+str(version)

X_dum, y_dum, X_all, y_all = misc.load_data(1)
X_all = rename_dir(X_all, 'train', vshowdir)
X_features,y_features = extract_features(X_all, y_all)

"""














""" feature_selection

version=15.300203201
showdir='show'
cv=10

vshowdir=showdir+'/v'+str(version)

X_dum, y_dum, X_all, y_all = misc.load_data(1)
X_all = rename_dir(X_all, 'train', vshowdir)
X_features,y_features = extract_features(X_all, y_all)
    
val_pct=1/cv
fold=1

X_train, y_train, X_val, y_val = misc.load_data(val_pct=val_pct, cv=fold)
X_train = rename_dir(X_train, 'train', vshowdir)
X_val = rename_dir(X_val, 'train', vshowdir)
X = []; y = []
for i in range(len(X_train)):
    if not X_train[i] in X_features: continue
    X.append(X_features[X_train[i]])
    y.append(y_features[y_train[i]])

Xv = []; yv = [];
for i in range(len(X_val)):
    if not X_val[i] in X_features: continue
    Xv.append(X_features[X_val[i]])
    yv.append(y_features[y_val[i]])


X_keep=[]
Xv_keep=[]
feats_keep=[]

feats=range(len(X[0]))
bestdice=0
bestfeat=-1
for feat in feats:
    improved=False
    for feat in feats:
        if feat in feats_keep: continue
        X_new=np.array([w[feat] for w in X]).reshape(-1, 1)
        Xv_new=np.array([w[feat] for w in Xv]).reshape(-1, 1)
        if len(X_keep) :
            X_new=np.hstack((X_keep,X_new))
            Xv_new=np.hstack((Xv_keep,Xv_new))
        # SVM
        clf = svm.SVC(kernel='linear')
        clf.fit(X_new,y)
        yp = clf.predict(Xv_new)
        #
        mean_dice = 0
        n = 0
        for i in range(len(X_val)):
            if not X_val[i] in X_features: continue
            imgdice = dice_predict(misc.load_image(X_val[i]) * yp[n],  misc.load_image(y_val[i]))
            mean_dice += imgdice
            n+=1
        mean_dice/=n
        print("feat "+str(feat)+" mean_dice: "+str(mean_dice))
        #
        if mean_dice < bestdice:
            bestfeat = feat
            bestdice = mean_dice
            print("best")
            improved = True
    if not improved:
        break
    else:
        feats_keep.append(bestfeat)
        X_new=np.array([w[bestfeat] for w in X]).reshape(-1, 1)
        Xv_new=np.array([w[bestfeat] for w in Xv]).reshape(-1, 1)
        if len(X_keep) :
            X_new=np.hstack((X_keep,X_new))
            Xv_new=np.hstack((Xv_keep,Xv_new))
        else:
            X_keep=X_new
            Xv_keep=Xv_new
"""








""" select variables 
gamma=0.001
C=0.1

# for g in range(-3,4):

for g in range(-6,-3):
    gamma=math.pow(10,g)
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_keep,y)
    yp = clf.predict(Xv_keep)
    #
    mean_dice = 0
    n = 0
    for i in range(len(X_val)):
        if not X_val[i] in X_features: continue
        imgdice = dice_predict(misc.load_image(X_val[i]) * yp[n],  misc.load_image(y_val[i]))
        mean_dice += imgdice
        n+=1
    mean_dice/=n
    print("gamma "+str(gamma)+" mean_dice: "+str(mean_dice))

for g in range(-3,4):
    c=math.pow(10,g)
    clf = svm.SVC(C=c)
    clf.fit(X_keep,y)
    yp = clf.predict(Xv_keep)
    #
    mean_dice = 0
    n = 0
    for i in range(len(X_val)):
        if not X_val[i] in X_features: continue
        imgdice = dice_predict(misc.load_image(X_val[i]) * yp[n],  misc.load_image(y_val[i]))
        mean_dice += imgdice
        n+=1
    mean_dice/=n
    print("C "+str(c)+" mean_dice: "+str(mean_dice))


gamma=0.001
C=0.1
clf = svm.SVC(gamma=gamma, C=C)
clf.fit(X_keep,y)
yp = clf.predict(Xv_keep)
#
mean_dice = 0
n = 0
for i in range(len(X_val)):
    if not X_val[i] in X_features: continue
    imgdice = dice_predict(misc.load_image(X_val[i]) * yp[n],  misc.load_image(y_val[i]))
    mean_dice += imgdice
    n+=1

mean_dice/=n
print("gamma "+str(gamma)+" mean_dice: "+str(mean_dice))



mean_dice = 0
n = 0
for i in range(len(X_val)):
    if not X_val[i] in X_features: continue
    p=misc.load_image(X_val[i]).sum()>=5000
    imgdice = dice_predict(misc.load_image(X_val[i]) * p,  misc.load_image(y_val[i]))
    mean_dice += imgdice
    n+=1

mean_dice/=n
print("orig dice: "+str(mean_dice))

"""
