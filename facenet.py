# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:21:24 2020

@author: Tsai Jen Kai
"""

import numpy as np
import cv2
import tensorflow as tf
sess = tf.Session()
import pickle
import signal
import matplotlib.pyplot as plt
import time
from IPython import display
from skimage.transform import resize
from tensorflow.keras.models import load_model
from keras import backend as K
K.set_session(sess)
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def calc_embs(imgs, margin, batch_size=1):
    aligned_images = prewhiten(imgs)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(facenet_model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

def capture_images(data,frame, name='Unknown'):
    
    margin = 10
    n_img_per_person = 200
    
    cascade = cv2.CascadeClassifier(cascade_path)
#        signal.signal(signal.SIGINT, self._signal_handler)

#        is_capturing, frame = vc.read()

    faces = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)
    if len(faces) != 0:
        cv2.imwrite('./data/'+name+'/'+str(len(imgs))+'.jpg', frame)
        face = faces[0]
        (x, y, w, h) = face
        left = x - margin // 2
        right = x + w + margin // 2
        bottom = y - margin // 2
        top = y + h + margin // 2
        img = resize(frame[bottom:top, left:right, :],
                     (160, 160), mode='reflect')
        imgs.append(img)
        cv2.rectangle(frame,
                      (left-1, bottom-1),
                      (right+1, top+1),
                      (255, 0, 0), thickness=2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        plt.title('{}/{}'.format(len(imgs), n_img_per_person))
        plt.xticks([])
        plt.yticks([])
        display.clear_output(wait=True)
        
        if len(imgs) == n_img_per_person:
#            vc.release()
            data[name] = np.array(imgs)

        try:
            plt.pause(0.1)
        except Exception:
            pass

    return data


def train(data):
    margin = 10
    batch_size = 1
    labels = []
    embs = []
    names = data.keys()
    for name, imgs in data.items():
        embs_ = calc_embs(imgs, margin, batch_size)    
        labels.extend([name] * len(embs_))
        embs.append(embs_)

    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    clf = SVC(kernel='linear', probability=True).fit(embs, y)
    
    'save clf'
    with open('trained_model/0720clf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    'save le'
    with open('trained_model/0720le.pkl', 'wb') as ff:
        pickle.dump(le, ff)

def infer(frame):
                       
    faces = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))
    pred = None
    if len(faces) == 1:
        face = faces[0]
        (x, y, w, h) = face
        left = x - margin // 2
        right = x + w + margin // 2
        bottom = y - margin // 2
        top = y + h + margin // 2
        img = resize(frame[abs(bottom):abs(top), abs(left):abs(right), :],
                     (160, 160), mode='reflect')
        #cv2.imshow('img',img)
        #cv2.waitKey(1)
        embs = calc_embs(img[np.newaxis], margin, 1)
        pred = le.inverse_transform(clf.predict(embs))
        
        cv2.rectangle(frame, (left-1, bottom-1), (right+1, top+1),(255, 0, 0), thickness=2)
        shframe = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
        cv2.putText(shframe,str(pred), (left-1,top-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow('facenet',shframe)
        cv2.waitKey(1)
            
    return pred

#f.capture_images('Chen_hsin_hung')
#f.train()
cascade_path = './model/cv2/haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascade_path)
model_path = './model/keras/model/facenet_keras.h5'
facenet_model = load_model(model_path)
margin = 10
imgs = []
'load model'
with open('trained_model/0720clf.pickle', 'rb') as fff:
    clf = pickle.load(fff)
with open('trained_model/0720le.pkl', 'rb') as ff:
    le = pickle.load(ff)

#init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#
#vc = cv2.VideoCapture(0)
#cv2.namedWindow('Real-time Multiple Person Action Recognition System', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Real-time Multiple Person Action Recognition System', 640, 480)
#with tf.Session() as sess:
#    sess.run(init)
#    
#    while(vc.isOpened()):
#        
#        rval , original_frame = vc.read()
#        if rval == True:
#    
#            t1 = time.time()
#    #        original_frame = original_frame[:,240:1680]
#            fframe = cv2.resize(original_frame,(640,480))
#            frame = fframe.copy()
#            frame = infer(frame)
#            cv2.imshow('face recognition', frame)
#            key = cv2.waitKey(1) & 0xFF
#            if key == ord('q'):
#                vc.release()
#                break
#        else:
#            print('no camera')
cv2.destroyAllWindows()