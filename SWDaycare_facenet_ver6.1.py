from fast_class1 import NMS_1d, Openvideo, SlidingWindow, load_pre_trained_model,\
                        get_class_labels, I3DVideoStackAppend\
                        
import facenet
import time
import cv2
import numpy as np
import tensorflow as tf
sess = tf.Session()
import i3d
import sys

#sys.path.append('D:/GitHub/darknet')
#import darknet
from PIL import Image
from utils import utils
from socket import *
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
#from deep_sort.detection import Detection as ddet

#'''tcpip parameter'''
#serverName = '127.0.0.1'
#serverPort = 50007

'''deepsort parameter'''
yolo = YOLO()
# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort 
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

'''Initial parameter'''
SAMPLE_FRAME = 16
SLIDING_WINDOW_LAYER = 1
MAXIMUM_FRAME = ( SAMPLE_FRAME * 2**(SLIDING_WINDOW_LAYER-1) )
MODE = 'video' # camera or video
out = cv2.VideoWriter('0318demo.avi',cv2.VideoWriter_fourcc(*'XVID'), 30.0, (1920,1080))
_SAMPLE_VIDEO_FRAMES = 16
_IMAGE_SIZE = 224
NUM_CLASSES = 12
_BATCH_SIZE = 1
FPS = 0.0
STEP = 4
zoomrate = 2.25
totalresult = []
ALLVideoStack = []
Picked_BBox = []
people = {}
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),
        (128,0,128),(128,128,0),(0,128,128),(128,128,128),
        (128,0,255),(128,255,0),(0,128,255),(255,128,0),(255,0,128),
        (255,128,128),(128,255,128),(128,128,255)]

classes_name = 'ClassDay2.0.list' 
_CHECKPOINT_PATHS = {'rgb_Day': 'models/rgb_model-318300'}

'''Definition of tensorflow'''
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
tf.logging.set_verbosity(tf.logging.INFO)
eval_type = FLAGS.eval_type
imagenet_pretrained = FLAGS.imagenet_pretrained

'''Placeholder of input images'''
labels_placeholder = tf.placeholder(tf.float32, [_BATCH_SIZE, NUM_CLASSES])
keep_prob = tf.placeholder(tf.float32)

'''Build the proposal network with i3d'''
# Proposal RGB input has 3 channels.
proposal_input = tf.placeholder(tf.float32,
    shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

with tf.variable_scope('RGB'):
  proposal_model = i3d.InceptionI3d( NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits' )
  proposal_logits, _ = proposal_model( proposal_input, is_training=True, dropout_keep_prob=keep_prob )

proposal_variable_map = {}
proposal_saver_savedata = {}
for variable in tf.global_variables():
  if variable.name.split('/')[0] == 'RGB':
     proposal_variable_map[variable.name.replace(':0', '')] = variable        
proposal_saver_savedata =  tf.train.Saver(var_list=proposal_variable_map, reshape=True)

model_logits = proposal_logits

'''Tensorflow output definition''' 
model_predictions = tf.nn.softmax(model_logits)
output_class = tf.argmax(model_predictions, 1)

'''Initialize the tensorflow variavles'''
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

'''Get the class labels'''
answer_names = get_class_labels(classes_name) 

#class_path='config/coco.names'
#classes = utils.load_classes(class_path)
frame_num = 0
cv2.namedWindow('Real-time Multiple Person Action Recognition System', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Real-time Multiple Person Action Recognition System', 640, 480)    
sess.run(init)
load_pre_trained_model(sess,  _CHECKPOINT_PATHS['rgb_Day'], proposal_saver_savedata)

with Openvideo(filename = r'0713original.mp4', mode = MODE) as vc:
    
    while(vc.isOpened()):
    
        rval , original_frame = vc.read()
        if rval == True:
            frame_num = frame_num + 1
            '''Timer start'''
    #           out.write(original_frame)
            
#            original_frame = original_frame[:,240:1680]
            fframe = cv2.resize(original_frame,(640,480))
#            frame = fframe.copy()
#            frame = infer(frame)
            ti1 = time.time()
            '''yolov3 + deepsort'''
            image = Image.fromarray(fframe[...,::-1]) #bgr to rgb                
            boxs = yolo.detect_image(image)
            features = encoder(fframe,boxs)
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            tracker.predict()
            tracker.update(detections)
            
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                #bbox ----->  (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])
                '對應座標'
                bbox = np.maximum(bbox, 0)
                ID = track.track_id
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                box_h = int(bbox[3]-y1)
                box_w = int(bbox[2]-x1)
                '1080原圖切人物'
                crop_person = original_frame[int(y1*zoomrate):int((y1+box_h)*zoomrate),
                                             int(x1*zoomrate):int((x1+box_w)*zoomrate)]
                
                'new ID ??'
                try:
                    'old ID'
                    personname = people[ID]
                except KeyError:
                    'have new person run facenet'
                    personname = facenet.infer(crop_person)
                    if personname!=None:
                        people[ID] = str(personname[0])
                    
                if( frame_num%STEP == 0):                   
                    
                    '''zoom in ?'''
                    if box_h > 160:
                        
                        '人物夠大直接做模糊'
                        back_frame = cv2.GaussianBlur(fframe, (0, 0), sigmaX=15, sigmaY=15)
                        _back_frame = back_frame.copy()
                        crop_person_s = fframe[y1:y1+box_h,x1:x1+box_w]
                        _back_frame[y1:y1+box_h,x1:x1+box_w] = crop_person_s
                        
                    else:
                        '人物太小'
                        '找人物中心點位置'
                        
                        center_x = int((x1+ 0.5*box_w)*zoomrate)
                        center_y = int((y1+ 0.5*box_h)*zoomrate)
    
                        '切出人物'
#                        crop_person_l = original_frame[int(y1*zoomrate):int((y1+box_h)*zoomrate),int(x1*zoomrate):int((x1+box_w)*zoomrate)]
                        
                        '以人物為中心切出480*640背景'
                        cropy1 = center_y-240
                        cropy2 = cropy1+480
                        cropx1 = center_x-320
                        cropx2 = cropx1+640
                        
                        '若超出邊界則對齊邊界'
                        if cropx1 < 0:
                            cropx1 = 0
                            cropx2 = 640
                            
                        if cropy1 < 0:
                            cropy1 = 0
                            cropy2 = 480
                            
                        if cropy2 > 1080:
                            cropy1 = 600
                            cropy2 = 1080
                            
                        if cropx2 > 1440:
                            cropx1 = 800
                            cropx2 = 1440
                            
                        '模糊背景'
                        ori_blur_frame = cv2.GaussianBlur(original_frame, (0, 0), sigmaX=15)
                        '人物貼到模糊背景'
                        ori_blur_frame[int(y1*zoomrate):int((y1+box_h)*zoomrate),int(x1*zoomrate):int((x1+box_w)*zoomrate)] = crop_person
                        '切出zoomin後辨識畫面'
                        zin_blur_frame = ori_blur_frame[cropy1:cropy2,cropx1:cropx2] 
                        
                        '儲存'
                        _back_frame = zin_blur_frame
                        
                        
                    '''Stack the video'''
                    try:
                        I3DVideoStackAppend(ALLVideoStack[ID].append, _back_frame)
                    except IndexError: 
                        for addspc in range (ID-len(ALLVideoStack)+1):
                            ALLVideoStack.append([])
                        I3DVideoStackAppend(ALLVideoStack[ID].append, _back_frame)
                            
                    if( len(ALLVideoStack[ID]) > MAXIMUM_FRAME ):
                        ALLVideoStack[ID].pop(0)
                    
                    '''Sliding window segmentation'''
                    with SlidingWindow(ALLVideoStack[ID], LayersOfScale=SLIDING_WINDOW_LAYER) as SlidingSegment:
                        
                        rgb_buffer = SlidingSegment.OutputStack
    
                        if SlidingSegment.State:
                                               
                            for slidinglayer in range(SlidingSegment.State):
                                
                                class_out = sess.run([model_predictions], feed_dict={proposal_input: rgb_buffer[slidinglayer], keep_prob: 1})          
                                a=list(class_out[0][0])
                                current_label = int(a.index(max(a)))
#                                print(current_label)
                                BBox_data = [(frame_num-((slidinglayer+1)*STEP*SAMPLE_FRAME)), frame_num, max(a),current_label]
                            try:
                                totalresult[ID].append(BBox_data)
                            except IndexError:
                                for ept in range (ID-len(totalresult)+1):
                                    totalresult.append([])
                                totalresult[ID].append(BBox_data)
                            
                            '1D NMS'
                            if( len(totalresult[ID]) == 7):
                                try:
                                    Picked_BBox[ID] = NMS_1d(totalresult[ID])
                                except IndexError:
                                    for apt in range (ID-len(Picked_BBox)+1):
                                        Picked_BBox.append([])
                                    Picked_BBox[ID] = NMS_1d(totalresult[ID])
                                totalresult[ID].clear()
        
#            'tcpoutput = [ID, action_index, Ix, Iy, ...x5...]'
#            tcpoutput = [0]*20
#            outputnumber = 0
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                #bbox ----->  (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])
                
                '對應座標'
                bbox = np.maximum(bbox, 0)
                ID = track.track_id
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                box_h = int(bbox[3]-y1)
                box_w = int(bbox[2]-x1)
                    
                try:
                    NMS_result_class = answer_names[Picked_BBox[ID][-1][3]]
#                    tcpoutput[outputnumber*4:outputnumber*4+4] = [ID, Picked_BBox[ID][-1][3], int((x1+(box_w/2))/3), int((y1+box_h)/3)]
#                    outputnumber = outputnumber+1
                except IndexError:
                    NMS_result_class = 'Background'
    
                color = colors[int(ID) % len(colors)]
                cv2.rectangle(fframe, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                
                try:
                    personname = people[ID]
                except KeyError:
                    personname = 'Unknown'
                    
                cv2.rectangle(fframe, (x1-2, y1-22), (x1+50 +(len(str(ID))+len(personname))*15, y1), color, -1)
                cv2.rectangle(fframe, (x1-2, y1-49), (x1 +len(NMS_result_class)*16, y1-27), color, -1)
                cv2.putText(fframe,'ID:'+str(ID)+' '+str(personname), (x1,y1-4),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(fframe, NMS_result_class, (x1,y1-31),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
            '''Timer end'''
            FPS = 1/(time.time()-ti1)
            cv2.putText(fframe, 'FPS:' + '{:5.1f}'.format(FPS), (20, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
    #                print(tcpoutput)
    #                clientSocket =socket(AF_INET,SOCK_STREAM)
    #                clientSocket.connect((serverName,serverPort))
    #                sentence = tcpoutput
    #                clientSocket.send(bytes(sentence))
            
            cv2.imshow('Real-time Multiple Person Action Recognition System', fframe)
    #                out.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            break

out.release()
cv2.destroyAllWindows()