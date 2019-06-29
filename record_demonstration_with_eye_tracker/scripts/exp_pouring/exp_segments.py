# Get color distribution of gaze points per segment (reach, grasp, pour etc)

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
parser.add_argument("-eid", type=str, default="1c", help='Experiment ID')
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
# bag_dir = '../../data/bags/'

if args.eid == '1c':
    print('Percentage of time during entire demo - spent on objects or other parts of workspace')
    print('Measure differences between novice and experts - video demos')

    init_color_dict = {
        'red': 0,
        'yellow': 0,
        'blue': 0,
        'green': 0,
        'black': 0,
        'other': 0,
        'pasta': 0
    }

    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all users
        print("processing Users' Video Demos...")
        for i in range(len(u)):
            user = u[i]
            print(user) #KT1,KT2
            dir_name = os.listdir(user_dir+user)

            a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
            d = os.listdir(a)

            exps = order[user]

            for seg in d:
                print('Segment ', seg)
                demo_type = exps[0] if int(seg)<=3 else exps[1]

                if(int(seg)!=1 and int(seg)!=4):
                    continue

                if demo_type!='v':
                    continue

                segment_colors = {
                    'Reaching_1': init_color_dict,
                    'Grasping_1': init_color_dict,
                    'Transport_1': init_color_dict,
                    'Pouring_1': init_color_dict,
                    'Return_1': init_color_dict,
                    'Release_1': init_color_dict,
                    'Reaching_2': init_color_dict,
                    'Grasping_2': init_color_dict,
                    'Transport_2': init_color_dict,
                    'Pouring_2': init_color_dict,
                    'Return_2': init_color_dict,
                    'Release_2': init_color_dict
                }

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)
                keyframe_indices = get_video_keyframes(user, video_file, video_kf_file)
                # print(keyframe_indices)
                start_idx, end_idx = keyframe_indices['Start'][0], keyframe_indices['Stop'][0]
                fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)
                


                video_file = a+seg+'/fullstream.mp4'
                # if(demo_type=='k'):
                #     keyframes, keyframe_indices = get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
                if(demo_type=='v'):
                    keyframes, keyframe_indices = get_video_keyframe_labels(user, video_file, video_kf_file)
                
                # Find end of first pouring - start of next pouring
                first_grasp = False
                pouring_round = 0

                hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)


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
                    segment_colors[kf_type+'_'+str(pouring_round+1)] = fixations


                all_fix[user[2:]] = segment_colors

    # print(all_expert_fix)
    # print(all_novice_fix)
    with open('1c_video_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[experts[0][2:]]['Reaching_1'].keys()
        u_color_names = ['User ID + Segment'] + color_names
        expert_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, expert_fix in all_expert_fix.items():
            for s in expert_fix.keys(): # segments
                value_list = [expert_fix[s][i] for i in color_names]
                value_list = [us+'+'+s] + value_list
                expert_writer.writerow(value_list)

    with open('1c_video_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[novices[0][2:]]['Reaching_1'].keys()
        u_color_names = ['User ID + Segment'] + color_names
        novice_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, novice_fix in all_novice_fix.items():
            for s in novice_fix.keys(): # segments
                value_list = [novice_fix[s][i] for i in color_names]
                value_list = [us+'+'+s] + value_list
                novice_writer.writerow(value_list)


# TODO: get center for blue and red blobs in image frame, then identify white gaze wrt that
# TODO: fix bug in KT color filtering per segment (video vis)
# TODO: birl - gaze term computed rbf wise versus object wise
if args.eid == '1d':
    print('Percentage of time during entire demo - spent on objects, gripper or other parts of workspace')
    print('Measure differences between novice and experts - KT demos')

    init_color_dict = {
        'red': 0,
        'yellow': 0,
        'blue': 0,
        'green': 0,
        'black': 0,
        'other': 0,
        'pasta': 0
    }

    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Users' KT Demos...")
        for i in range(len(u)):
            user = u[i]
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

                if demo_type!='k':
                    continue

                segment_colors = {
                    'Reaching_1': init_color_dict,
                    'Grasping_1': init_color_dict,
                    'Transport_1': init_color_dict,
                    'Pouring_1': init_color_dict,
                    'Return_1': init_color_dict,
                    'Release_1': init_color_dict,
                    'Reaching_2': init_color_dict,
                    'Grasping_2': init_color_dict,
                    'Transport_2': init_color_dict,
                    'Pouring_2': init_color_dict,
                    'Return_2': init_color_dict,
                    'Release_2': init_color_dict
                }

                bag_file = ''
                for file in bagfiles:
                    # print(file)
                    if (file.endswith("kt-p1.bag") and (int(seg)==1 or int(seg)==4)):
                        bag_file = bagloc + file
                    elif (file.endswith("kt-p2.bag") and (int(seg)==2 or int(seg)==5)):
                        bag_file = bagloc + file
                    elif (file.endswith("kt-p3.bag") and (int(seg)==3 or int(seg)==6)):
                        bag_file = bagloc + file

                
                if bag_file == '':
                    # print(user, seg)
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                
                keyframes, keyframe_indices = get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
                
                first_grasp = False
                pouring_round = 0

                hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)

                start_idx = 0   
                for fid in keyframe_indices:
                    kf_type = keyframes[fid]
                    if(kf_type=='Open'):
                        first_grasp = True
                    if kf_type=='Reaching' and first_grasp:
                        pouring_round = 1
                    end_idx = fid
                    fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)
                    segment_colors[kf_type+'_'+str(pouring_round+1)] = fixations

                    if kf_type=='Open' or kf_type=='Close':
                        kf_type = 'Grasping'
                    if kf_type not in target_objects.keys():
                        start_idx = end_idx
                        continue
                    start_idx = end_idx

                all_fix[user[2:]] = segment_colors

                

    with open('1d_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)       
        color_names = all_expert_fix[all_expert_fix.keys()[0]]['Reaching_1'].keys()
        u_color_names = ['User ID + Segment'] + color_names
        expert_writer.writerow(u_color_names)
        for us, expert_fix in all_expert_fix.items():
            for s in expert_fix.keys(): # segments
                value_list = [expert_fix[s][i] for i in color_names]
                value_list = [us+'+'+s] + value_list
                expert_writer.writerow(value_list)

    with open('1d_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[all_novice_fix.keys()[0]]['Reaching_1'].keys()
        u_color_names = ['User ID + Segment'] + color_names
        novice_writer.writerow(u_color_names)
        for us, novice_fix in all_novice_fix.items():
            for s in novice_fix.keys(): # segments
                value_list = [novice_fix[s][i] for i in color_names]
                value_list = [us+'+'+s] + value_list
                novice_writer.writerow(value_list)