from utils import takeClosest, get_hsv_color_timeline, get_color_name_from_hist, get_kt_keyframes_labels
from utils import get_video_keyframes, read_json, filter_fixations, get_kt_keyframes, get_video_keyframe_labels
from utils import filter_fixations_with_timeline, get_step_kf_indices, filter_fixations_ignore_black, filter_fixation_counts
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import math
import cv2

from numpy import ones,vstack
from numpy.linalg import lstsq
import ast 
import gzip
from utils import get_hsv_color_timeline, get_color_name_from_hist


from bisect import bisect_left
def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

expert_dir = '../../data/pouring/experts/'    
experts = os.listdir(expert_dir)

order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

condition_names = {
    'k': 'KT demo',
    'v': 'Video demo'
}

video_kf_file = 'video_kf.txt'
bag_dir = '/home/akanksha/Documents/gaze_for_lfd_study_data/gaze_lfd_user_study/'

kt_target_acc, video_target_acc = {}, {}

target_objects = {
'Reaching': [['green', 'pasta'], ['yellow', 'pasta']],
'Grasping': [['green', 'pasta'], ['yellow', 'pasta']],
'Transport': [['red'], ['blue']],
'Pouring': [['red', 'pasta'],['blue','pasta']],
'Return': [['other'],['other']],
'Release': [['green'],['yellow']]
}
user_dir = expert_dir
print("processing Expert Users' Video Demos...")
# for i in range(len(experts)):
# for i in range(1):
user = experts[1]
print(user) #KT1,KT2
dir_name = os.listdir(user_dir+user)

a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
d = os.listdir(a)

exps = order[user]

bagloc = bag_dir + user + '/bags/'
bagfiles = os.listdir(bagloc)

# for seg in d:
seg = '1'
print('Segment ', seg)
demo_type = exps[0] if int(seg)<=3 else exps[1]
print(demo_type)
# if(int(seg)!=1 and int(seg)!=4):
#     continue

# if demo_type!='k':
#     continue

bag_file = ''
if(demo_type=='k'):
    for file in bagfiles:
        # if (file.endswith("kt-p1.bag")):
        #     bag_file = bagloc + file
        if (file.endswith("kt-p1.bag") and (int(seg)==1 or int(seg)==4)):
            bag_file = bagloc + file
        # elif (file.endswith("kt-p2.bag") and (int(seg)==2 or int(seg)==5)):
        #     bag_file = bagloc + file
        # elif (file.endswith("kt-p3.bag") and (int(seg)==3 or int(seg)==6)):
        #     bag_file = bagloc + file
    
    if bag_file == '':
        print('Bag file does not exist for KT demo, skipping...')
        exit()
        # continue

target_acc = {
    'Reaching': [0, 0],
    'Grasping': [0, 0],
    'Transport': [0, 0],
    'Pouring': [0, 0],
    'Return': [0, 0],
    'Release': [0, 0]
}

detected_ref_frame = {
    'Reaching': ['other', 'other'],
    'Grasping': ['other', 'other'],
    'Transport': ['other', 'other'],
    'Pouring': ['other', 'other'],
    'Return': ['other', 'other'],
    'Release': ['other', 'other']
}

color_dict = {
    'black': (0,0,0),
    'red': (0,0,255),
    'green': (0,255,0),
    'yellow': (0,255,255),
    'pasta': (0,255,255),
    'blue': (255,0,0),
    'other': (192,192,192)
}

data, gp, model, all_vts = read_json(a+seg)
video_file = a+seg+'/fullstream.mp4'
if(demo_type=='k'):
    keyframes, keyframe_indices = get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
if(demo_type=='v'):
    keyframes, keyframe_indices = get_video_keyframe_labels(user, video_file, video_kf_file)
# print(keyframes)
# Find end of first pouring - start of next pouring
first_grasp = False
pouring_round = 0

hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)



if(demo_type=='k'):
    start_idx = 0   
    for fid in keyframe_indices:
        kf_type = keyframes[fid]
        if(kf_type=='Open'):
            first_grasp = True
        if kf_type=='Reaching' and first_grasp:
            pouring_round = 1
        end_idx = fid
        fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)
        

        if kf_type=='Open' or kf_type=='Close':
            kf_type = 'Grasping'
        if kf_type not in target_objects.keys():
            start_idx = end_idx
            continue

        # assign max value to the color of the default target of this KF
        max_val = 0
        for o in target_objects[kf_type][pouring_round]:
            # print(o)
            if(fixations[o]!=-1):
                max_val += fixations[o]
        max_color = target_objects[kf_type][pouring_round][0]
        for key, val in fixations.items():
            if(val==-1): #bug
                continue
            if val>max_val:
                max_val = val
                max_color = key
        if(max_val>0):
            if max_color == target_objects[kf_type][pouring_round][0]:
                # target_acc[kf_type][0] += 1
                detected_ref_frame[kf_type][pouring_round] = max_color
            # target_acc[kf_type][1] += 1
        # all_fix.append(fixations)
        # One plot showing both novice and expert numbers for objects, other
        start_idx = end_idx
    # kt_target_acc[user[2:]+'_'+str(seg)] = target_acc

if(demo_type=='v'):
    # start_idx = 0
    # print(keyframe_indices) 
    for j in range(1,len(keyframe_indices)-1):
        fid = keyframe_indices[j]
        kf_type = keyframes[fid]
        # print(kf_type)
        if(kf_type=='Pouring'):
            first_grasp = True
        if kf_type=='Reaching' and first_grasp:
            pouring_round = 1
        start_idx = fid
        end_idx = keyframe_indices[j+1]
        fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

        # assign max value to the color of the default target of this KF
        max_val = 0
        for o in target_objects[kf_type][pouring_round]:
            # print(o)
            if(fixations[o]!=-1):
                max_val += fixations[o]
        max_color = target_objects[kf_type][pouring_round][0]
        for key, val in fixations.items():
            if(val==-1):
                continue
            if val>max_val:
                max_val = val
                max_color = key
        if(max_val>0):
            if max_color == target_objects[kf_type][pouring_round][0]:
                # target_acc[kf_type][0] += 1
                detected_ref_frame[kf_type][pouring_round] = max_color
        print(max_color)
            # target_acc[kf_type][1] += 1
        # all_fix.append(fixations)
        # One plot showing both novice and expert numbers for objects, other
        # start_idx = end_idx
    # print(experts[0][2])
    # video_target_acc[user[2:]+'_'+str(seg)] = target_acc


print(dir_name)
print(a)
data_file = a+seg+"/livedata.json.gz"
with gzip.open(data_file, "rb") as f:
    data=f.readlines()

    for r in range(len(data)):
        row = data[r]
        data[r] = ast.literal_eval(row.strip('\n'))

vid2ts = {}     # dictionary mapping video time to time stamps in json
right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

for d in data:
    if 'vts' in d and d['s']==0:
        if d['vts'] == 0:
            vid2ts[d['vts']] = d['ts']
        else:
            vid2ts[d['vts']] = d['ts']

    # TODO: if multiple detections for same time stamp?
    if 'pd' in d and d['s']==0 and d['eye']=='right':
        right_eye_pd[d['ts']] = d['pd']
    if 'pd' in d and d['s']==0 and d['eye']=='left':
        left_eye_pd[d['ts']] = d['pd']

    if 'gp' in d and d['s']==0 :
        gp[d['ts']] = d['gp']   #list of 2 coordinates
print('read json')

vidcap = cv2.VideoCapture(video_file)
fps = vidcap.get(cv2.CAP_PROP_FPS)
# print fps   #25 fps
success, img = vidcap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('../../data/RSS videos/pouring_ref_frame_3.avi',fourcc,fps,(1920,1080))

all_ts = sorted(gp.keys())
count = 0
frame2ts = []
pouring_round = 0
font = cv2.FONT_HERSHEY_SIMPLEX
first_grasp = False
while success:  
    # print(count)
    frame_ts = int((count/fps)*1000000)
    frame2ts.append(frame_ts)

    less = [a for a in all_vts if a<=frame_ts]
    idx = len(less)-1

    if idx<len(model):
        m,c = model[idx]
    else:
        m,c = model[len(model)-1]
    ts = m*frame_ts + c

    tracker_ts = takeClosest(all_ts,ts)

    gaze = gp[tracker_ts]
    gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))

    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv = img_hsv[gaze_coords[1]][gaze_coords[0]]
    
    # color_name, color_value = get_color_name(hsv)
    radius = 100
    color_name, color_value = get_color_name_from_hist(gaze_coords, img_hsv, radius)
    # window.append(color_name)
    # if(len(window)>win_size):
    #     del window[0]

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # if(count in saccade_indices):
    #     # cv2.putText(img, 'SACCADE!!', (800, 250), font, 1.8, (192,192,192), 5, cv2.LINE_AA)
    #     print 'saccade'
    # else:
    #     # might be a fixation
    #     fixation = True
    #     for det_c in window:
    #         if det_c!=color_name:
    #             fixation=False
    #     if(fixation):
    #         cv2.putText(img, '*****FIXATION****', (600, 150), font, 1.8, (0,0,0), 8, cv2.LINE_AA)
    #         cv2.putText(img, '*****FIXATION****', (600, 150), font, 1.8, color_value, 5, cv2.LINE_AA)
    #         cv2.putText(img, object_name[color_name], (700, 250), font, 1.8, (0,0,0), 8, cv2.LINE_AA)
    #         cv2.putText(img, object_name[color_name], (700, 250), font, 1.8, color_value, 5, cv2.LINE_AA)
        

    cv2.circle(img,gaze_coords, 25, (255,255,0), 3)
    cv2.circle(img,gaze_coords, radius, (0,165,255), 3)
    
    
    # print(keyframe_indices)
    if count in keyframe_indices:
        print('keyframe here!')
        idx = keyframe_indices.index(count)
        
        if idx!=len(keyframe_indices)-1:
            kf_type = keyframes[keyframe_indices[idx+1]]
        elif (keyframes[keyframe_indices[idx]]=='Grasping'):
            kf_type = 'Grasping'
        elif (keyframes[keyframe_indices[idx]]=='Reaching'):
            kf_type = 'Reaching'
        else:
            kf_type = keyframes[keyframe_indices[idx]]
            

    if(kf_type=='Pouring'):
            first_grasp = True
    if kf_type=='Reaching' and first_grasp:
        pouring_round = 1


    if kf_type=='Open' or kf_type=='Close':
        kf_type = 'Grasping'
    # print(kf_type)
    # print(pouring_round)
    # print(detected_ref_frame[kf_type][pouring_round])
    color_t = color_dict[target_objects[kf_type][pouring_round][0]]
    color_d = color_dict[detected_ref_frame[kf_type][pouring_round]]
    cv2.putText(img, '*****'+kf_type+'****', (600, 150), font, 1.8, (0,0,0), 8, cv2.LINE_AA)  
    cv2.putText(img, '*****'+kf_type+'****', (600, 150), font, 1.8, (255,255,255), 5, cv2.LINE_AA) 
    cv2.putText(img, 'Target Object: '+target_objects[kf_type][pouring_round][0], (650, 200), font, 1.0, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, 'Target Object: '+target_objects[kf_type][pouring_round][0], (650, 200), font, 1.0, color_t, 2, cv2.LINE_AA)
    cv2.putText(img, 'Detected Object: '+detected_ref_frame[kf_type][pouring_round], (650, 250), font, 1.0, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, 'Detected Object: '+detected_ref_frame[kf_type][pouring_round], (650, 250), font, 1.0, color_d, 2, cv2.LINE_AA)

    video.write(img)
    # cv2.imwrite('video_imgs/'+str(count)+'.png', img)
    count += 1
    success, img = vidcap.read()

cv2.destroyAllWindows()
video.release()