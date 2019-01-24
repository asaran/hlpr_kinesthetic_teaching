import argparse
from utils import takeClosest, get_hsv_color_timeline, get_color_name_from_hist, get_kt_keyframes_labels
from utils import get_video_keyframes, read_json, filter_fixations, get_kt_keyframes, get_video_keyframe_labels
from utils import filter_fixations_with_timeline, get_step_kf_indices, filter_fixations_ignore_black, filter_fixation_counts
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import math

parser = argparse.ArgumentParser()
parser.add_argument("-eid", type=str, default="1a", help='Experiment ID')
args = parser.parse_args()

expert_dir = '../../data/pouring/experts/'    
experts = os.listdir(expert_dir)
# print(users)

novice_dir = '../../data/pouring/novices/'    
novices = os.listdir(novice_dir)

order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

condition_names = {
    'k': 'KT demo',
    'v': 'Video demo'
}

video_kf_file = 'video_kf.txt'
bag_dir = '/home/akanksha/Documents/gaze_for_lfd_study_data/gaze_lfd_user_study/'

if args.eid == '1a':
    print('Do long fixations line up with keyframes?')
    print('Video versus KT demos (expert users)')

    plt.figure(1, figsize=(20,5))
    plt.figure(2, figsize=(20,5))

    # kt_target_acc, video_target_acc = {}, {}

    user_dir = expert_dir
    print("processing Expert Users' Video Demos...")
    changes_k, changes_v = {}, {}

    for i in range(len(experts)):
    # for i in range(1):
        user = experts[i]
        print(user) #KT1,KT2
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)
        
        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=3 else exps[1]

            if(int(seg)!=1 and int(seg)!=4):
                continue

            bag_file = ''
            if(demo_type=='k'):
                for file in bagfiles:
                    if (file.endswith("kt-p1.bag")):
                        bag_file = bagloc + file
                
                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue


            data, gp, model, all_vts = read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            if(demo_type=='k'):
                keyframes, keyframe_indices = get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
            if(demo_type=='v'):
                keyframes, keyframe_indices = get_video_keyframe_labels(user, video_file, video_kf_file)

            first_grasp = False
            pouring_round = 0

            hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)

            if(demo_type=='k'):
                start_idx = 0  
                end_idx = keyframe_indices[-1] 
                fixation_color_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

         
                c_prev  = fixation_color_list[0]
                changes, total = 0, 0
                for idx,c in enumerate(fixation_color_list):
                	if idx==0:
                		continue
                	if c!=c_prev:
                		changes += 1
                	total += 1
                	c_prev = c
                changes_k[user[2:]] = [changes, total]


            if(demo_type=='v'):
                start_idx = keyframe_indices[0] 
                end_idx = keyframe_indices[-2] 
                fixation_color_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

                c_prev  = fixation_color_list[0]
                changes, total = 0, 0
                for idx,c in enumerate(fixation_color_list):
                	if idx==0:
                		continue
                	if c!=c_prev:
                		changes += 1
                	total += 1
                	c_prev = c
                changes_v[user[2:]] = [changes, total]


    with open('kt_clean_experts.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        users_ = changes_k.keys()
        kf_names = ['changes', 'total']
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in changes_k.items():
            # value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [acc[j] for j in [0,1]]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)

    with open('video_clean_experts.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        users_ = changes_v.keys()
        kf_names = ['changes', 'total']
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in changes_v.items():
            # value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [acc[j] for j in [0,1]]
            value_list = [u] + value_list
            expert_writer.writerow(value_list) 



if args.eid == '1b':
    print('Do long fixations line up with keyframes?')
    print('Video versus KT demos (expert users)')

    plt.figure(1, figsize=(20,5))
    plt.figure(2, figsize=(20,5))

    # kt_target_acc, video_target_acc = {}, {}

    user_dir = novice_dir
    print("processing Novice Users' Video Demos...")
    changes_k, changes_v = {}, {}

    for i in range(len(novices)):
    # for i in range(1):
        user = novices[i]
        print(user) #KT1,KT2
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)
        
        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=3 else exps[1]

            if(int(seg)!=1 and int(seg)!=4):
                continue

            bag_file = ''
            if(demo_type=='k'):
                for file in bagfiles:
                    if (file.endswith("kt-p1.bag")):
                        bag_file = bagloc + file
                
                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue


            data, gp, model, all_vts = read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            if(demo_type=='k'):
                keyframes, keyframe_indices = get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
            if(demo_type=='v'):
                keyframes, keyframe_indices = get_video_keyframe_labels(user, video_file, video_kf_file)

            first_grasp = False
            pouring_round = 0

            hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)

            if(demo_type=='k'):
                start_idx = 0  
                end_idx = keyframe_indices[-1] 
                fixation_color_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

         
                c_prev  = fixation_color_list[0]
                changes, total = 0, 0
                for idx,c in enumerate(fixation_color_list):
                	if idx==0:
                		continue
                	if c!=c_prev:
                		changes += 1
                	total += 1
                	c_prev = c
                changes_k[user[2:]] = [changes, total]


            if(demo_type=='v'):
                start_idx = keyframe_indices[0] 
                end_idx = keyframe_indices[-2] 
                fixation_color_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

                c_prev  = fixation_color_list[0]
                changes, total = 0, 0
                for idx,c in enumerate(fixation_color_list):
                	if idx==0:
                		continue
                	if c!=c_prev:
                		changes += 1
                	total += 1
                	c_prev = c
                changes_v[user[2:]] = [changes, total]


    with open('kt_clean_novices.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        users_ = changes_k.keys()
        kf_names = ['changes', 'total']
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in changes_k.items():
            # value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [acc[j] for j in [0,1]]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)

    with open('video_clean_novices.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        users_ = changes_v.keys()
        kf_names = ['changes', 'total']
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in changes_v.items():
            # value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [acc[j] for j in [0,1]]
            value_list = [u] + value_list
            novice_writer.writerow(value_list) 