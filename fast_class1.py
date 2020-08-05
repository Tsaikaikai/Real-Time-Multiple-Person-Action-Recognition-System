# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:27:48 2019

@author: Tsai Jen Kai
"""
from models import *
from utils import *
import os, sys, time, datetime, random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from numba import autojit 
from torch.autograd import Variable

_SAMPLE_VIDEO_FRAMES = 16
_IMAGE_SIZE = 224

import matplotlib.pyplot as plt
import signal
from IPython import display
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from keras.models import load_model
import pickle

model_path = './model/keras/model/facenet_keras.h5'
facenet_model = load_model(model_path)

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

def calc_embs(imgs, margin, batch_size):
    aligned_images = prewhiten(imgs)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(facenet_model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

class FaceDemo(object):
    def __init__(self, cascade_path):

        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.margin = 10
        self.batch_size = 1
        self.n_img_per_person = 200
        self.is_interrupted = False
        self.data = {}
        
    def _signal_handler(self, signal, frame):
        self.is_interrupted = True
        
    def infer(self, frame):
        
        'load model'
        with open('trained_model/2pclf.pickle', 'rb') as f:
            clf = pickle.load(f)
        with open('trained_model/2ple.pkl', 'rb') as ff:
            le = pickle.load(ff)
            
        signal.signal(signal.SIGINT, self._signal_handler)
        self.is_interrupted = False

#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        faces = self.cascade.detectMultiScale(frame,
#                                     scaleFactor=1.1,
#                                     minNeighbors=3,
#                                     minSize=(100, 100))
        faces = self.cascade.detectMultiScale(frame,
                             scaleFactor=1.1,
                             minNeighbors=3)
        pred = None
        if len(faces) != 0:
            for face in faces:
                (x, y, w, h) = face
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                img = resize(frame[abs(bottom):abs(top), abs(left):abs(right), :],
                             (160, 160), mode='reflect')
                embs = calc_embs(img[np.newaxis], self.margin, 1)
                
                pred = le.inverse_transform(clf.predict(embs))                
                print(pred)
                cv2.rectangle(frame, (left-1, bottom-1), (right+1, top+1),(255, 0, 0), thickness=2)
                cv2.putText(frame,str(pred), (left-1,top-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
#                cv2.imshow('face recognition', frame)
        return frame

#load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'


def Take1Element(a):
    return a[0]

def Take3Element(n):
    return n[2]

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects


@autojit
def NMS_1d(BBox, threshold = 0.3):
    '''
        BBox = [ start, end, confidence, label ]
    '''
    if len(BBox) == 0:
        return []
    
    BBox.sort( key = Take3Element )
    BBox_array = np.array( BBox )
    
    picked_BBox = []
    
    startT = BBox_array[:,0]
    endT = BBox_array[:,1]
    confidence = BBox_array[:,2]
    
    areas = endT - startT
    
    '''Confidence score sort'''
    order = np.argsort(confidence)
    
    while order.size > 0:
        '''The index of largest confidence score'''
        index = order[-1]
        
        picked_BBox.append(BBox[index])
        
        t1 = np.maximum(startT[index], startT[order[:-1]])
        t2 = np.minimum(endT[index], endT[order[:-1]])
        
        IoU_area = np.maximum(0.0, t2-t1)
        
        ratio = IoU_area/(areas[index] + areas[order[:-1]] - IoU_area)
        
        left = np.where(ratio < threshold)
        order = order[left]
    
    picked_BBox.sort( key = Take1Element )
    return picked_BBox

@autojit
def load_pre_trained_model(sess, Path, rgb_saver_savedata):
    rgb_saver_savedata.restore(sess, Path)
    print('RGB checkpoint restored')
    print('RGB data loaded')
      
@autojit   
def get_class_labels(file_name):
    answer_names = []
    classInd_lines = open(file_name, 'r')
    classInd_list = list(classInd_lines)
    for index in range(len(classInd_list)) :
        answer = classInd_list[index].strip('\n').split()
        answer_names.append(answer[0])
        
    return answer_names 

def I3DVideoStackAppend(StackFun, image, ImageSize=224):
    img = Image.fromarray(image.astype(np.uint8))
    if(img.width>img.height):
        scale = float(ImageSize)/float(img.height)
        img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), ImageSize))).astype(np.float32)
    else:
        scale = float(ImageSize)/float(img.width)
        img = np.array(cv2.resize(np.array(img),(ImageSize, int(img.height * scale + 1)))).astype(np.float32)
    crop_x = int((img.shape[0] - ImageSize)/2)
    crop_y = int((img.shape[1] - ImageSize)/2)
    img = img[crop_x:crop_x+ImageSize, crop_y:crop_y+ImageSize,:]
    StackFun(img)


class Openvideo(object):
    def __init__(self, filename = '', mode = 'video'):
        self.filename = filename
        self.mode = mode
        
    def __enter__(self):
        if self.mode == 'camera':
#            self.cap = cv2.VideoCapture(cv2.CAP_DSHOW)
#            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            self.cap = cv2.VideoCapture(0)
            # 設定擷取影像的尺寸大小
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            return self.cap
        elif self.mode == 'video':
            print('Read video: ' + self.filename)
            self.openfile = cv2.VideoCapture('./' + self.filename)
            return self.openfile
        else:
            print('Fail to open video or camera.')
            return None
    
    def __exit__(self, type, value, traceback):
        if self.mode == 'camera':
            print('Close camera')
            self.cap.release()
        else:
            print('Close video: ' + self.filename)
        cv2.destroyAllWindows()
        

'''
    Class for tensorflow to generate the segments of sliding window
    
    Example:
    |================| Video stream
    |              --| 1st segment (OutputStack[0]) x1
    |            ----| 2nd segment (OutputStack[1]) x2
    |        --------| 3rd segment (OutputStack[2]) x4
    |----------------| 4th segment (OutputStack[3]) x8
'''
class SlidingWindow(object):
    """
        Initialize all parameters from start
            __ForeStack : Local variable, store a sequence of video
            OutputStack : Return variable, segments of sliding window
            InputStack : Input video array
            MinFrame : Parameter of frame size
            LayersOfScale : Number of sliding window layers
            State : Number of segments
    """
    @autojit
    def __init__(self, ImageStackInput, MinFrame=16, LayersOfScale=3):
        self.__ForeStack = []
        self.OutputStack = []
        self.InputStack = ImageStackInput
        self.MinFrame = MinFrame
        self.LayersOfScale = LayersOfScale
        self.State = 0
        self.__ForeStackInsert = self.__ForeStack.insert
        self.__OutputStackAppend = self.OutputStack.append
        
    @autojit
    def __enter__(self):
        
        for layer in range(self.LayersOfScale ):
            if(len(self.InputStack) >= (2**layer * self.MinFrame) ):
                """(Change)Read the memory address"""
                for index in range(-1, (-self.MinFrame * 2**layer - 1), -2**layer):
                    self.__ForeStackInsert(0, self.InputStack[index])   
                
                self.__OutputStackAppend( np.array([self.__ForeStack]).astype(np.float32) )
                self.__ForeStack.clear()
                self.State = self.State + 1
            else:
                break
        return self
    
    def __exit__(self, type, value, traceback):
        pass