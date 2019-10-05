# Experiments for the Pouring Task
# 1. Percentage of time during entire demo - spent on objects, gripper or other parts of workspace 
#TODO: get_hsv_color_timeline not using filter_fixations!!!
#       a. Measure differences between novice and experts - video demos
#       b. Measure differences between novice and experts - KT demos
# 2. Perecentage accuarcy to predict reference frame per keyframe (same consecutive keyframes clubbed together) -- NO ANOVA
#       a. Video versus KT demos (expert users)
#       b. Video versus KT demos (novice users)     
#       c. Video demos over time (expert users)
#       d. Video demos over time (novice users)
#       e. KT demos over time (expert users)
#       f. KT demos over time (novice users)
# 3. Do long fixations line up with Keyframes? Visualization of fixations across users.
#       a. Video versus KT demos (expert users)
#       b. Video versus KT demos (novice users)
# 4. Do keyframe and video demos match in overall fixations (removing black pixels, and tie break to next major color)?
# Chi-square test:   http://www.stat.yale.edu/Courses/1997-98/101/chigf.htm
# TODO: account for pasta with the right color object based on keyframe
#       a. Video versus KT demos (expert users)
#       b. Video versus KT demos (novice users)
# 5. Is gaze-based fixation frame different between step KF and non-step KF? 
#       a. KT demos - ref frame before and after KF (expert users)
#       b. KT demos - ref frame before and after KF (novice users)



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
# bag_dir = '../../data/bags/'

if args.eid == '1a':
    print('Percentage of time during entire demo - spent on objects or other parts of workspace')
    print('Measure differences between novice and experts - video demos')

    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Expert Users' Video Demos...")
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

                # if(int(seg)!=1 and int(seg)!=4):
                #     continue

                if demo_type!='v':
                    continue

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)
                keyframe_indices = get_video_keyframes(user, video_file, video_kf_file)
                # print(keyframe_indices)
                start_idx, end_idx = keyframe_indices['Start'][0], keyframe_indices['Stop'][0]
                fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)
                # all_fix.append(fixations)
                all_fix[user[2:]+'_'+str(seg)] = fixations
            # One plot showing both novice and expert numbers for objects, other

    # print(all_expert_fix)
    # print(all_novice_fix)
    with open('1a_video_expert_3trials.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[experts[0][2:]+'_1'].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('1a_video_novice_3trials.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[novices[0][2:]+'_1'].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)

if args.eid == '1b':
    print('Percentage of time during entire demo - spent on objects, gripper or other parts of workspace')
    print('Measure differences between novice and experts - KT demos')
    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Expert Users' Video Demos...")
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

                # if(int(seg)!=1 and int(seg)!=4):
                #     continue

                if demo_type!='k':
                    continue

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
                keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
                # print(keyframes)
                start_idx, end_idx = keyframes[0], keyframes[-1]
                hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)

                fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx,end_idx)
                # all_fix.append(fixations)
                all_fix[user[2:]+'_'+str(seg)] = fixations
                # print(user[2:]+'_'+str(seg))
                # One plot showing both novice and expert numbers for objects, other

    # print(all_fix)
    with open('1b_kt_expert_3trials.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # color_names = all_expert_fix[experts[0][2:]+'_4'].keys()
        color_names = all_expert_fix[all_expert_fix.keys()[0]].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('1b_kt_novice_3trials.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # color_names = all_novice_fix[novices[0][2:]+'_1'].keys()
        color_names = all_novice_fix[all_novice_fix.keys()[0]].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)

# TODO: Run 2a again as novices not evaluated
if args.eid == '2a':
    print('Perecentage accuarcy to predict reference frame per keyframe')
    print('Video versus KT demos (expert users)')

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
                    elif (file.endswith("kt-p2.bag") and (int(seg)==2 or int(seg)==5)):
                        bag_file = bagloc + file
                    elif (file.endswith("kt-p3.bag") and (int(seg)==3 or int(seg)==6)):
                        bag_file = bagloc + file
                
                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

            target_acc = {
                'Reaching': [0, 0],
                'Grasping': [0, 0],
                'Transport': [0, 0],
                'Pouring': [0, 0],
                'Return': [0, 0],
                'Release': [0, 0]
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
                    

                    if kf_type=='Open':
                        kf_type = 'Release'
                    if kf_type=='Close':
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
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1
                    # all_fix.append(fixations)
                    # One plot showing both novice and expert numbers for objects, other
                    start_idx = end_idx
                kt_target_acc[user[2:]+'_'+str(seg)] = target_acc

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
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1
                    # all_fix.append(fixations)
                    # One plot showing both novice and expert numbers for objects, other
                    # start_idx = end_idx
                # print(experts[0][2])
                video_target_acc[user[2:]+'_'+str(seg)] = target_acc

    # print(all_fix)
    with open('2a_kt_expert_3trials.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[experts[0][2:]].keys()
        kf_names = kt_target_acc[kt_target_acc.keys()[0]].keys()
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        
        # no_of_colors = length(color_names)
        for u,acc in kt_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1  for i in kf_names]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)

    with open('2a_video_expert_3trials.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = video_target_acc[experts[0][2:]].keys()
        kf_names = video_target_acc[video_target_acc.keys()[0]].keys()
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u,acc in video_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)
    

if args.eid == '2b':
    print('Perecentage accuarcy to predict reference frame per keyframe')
    print('Video versus KT demos (novice users)')
    
    kt_target_acc, video_target_acc = {}, {}

    target_objects = {
        'Reaching': [['green', 'pasta'], ['yellow', 'pasta']],
        'Grasping': [['green', 'pasta'], ['yellow', 'pasta']],
        'Transport': [['red'], ['blue']],
        'Pouring': [['red', 'pasta'],['blue','pasta']],
        'Return': [['other'],['other']],
        'Release': [['green'],['yellow']]
    }
    user_dir = novice_dir
    print("processing Expert Users' Video Demos...")
    for i in range(len(novices)):
    # for i in [4]:
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
                    elif (file.endswith("kt-p2.bag") and (int(seg)==2 or int(seg)==5)):
                        bag_file = bagloc + file
                    elif (file.endswith("kt-p3.bag") and (int(seg)==3 or int(seg)==6)):
                        bag_file = bagloc + file
                

                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

            target_acc = {
                'Reaching': [0, 0],
                'Grasping': [0, 0],
                'Transport': [0, 0],
                'Pouring': [0, 0],
                'Return': [0, 0],
                'Release': [0, 0]
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
                        if(val==-1):
                            continue
                        if val>max_val:
                            max_val = val
                            max_color = key

                    if(max_val>0):
                        if max_color == target_objects[kf_type][pouring_round][0]:
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1
                    # all_fix.append(fixations)
                    # One plot showing both novice and expert numbers for objects, other
                    start_idx = end_idx
                # kt_target_acc.append(target_acc)
                kt_target_acc[user[2:]+'_'+str(seg)] = target_acc

            if(demo_type == 'v'):
                # start_idx = 0
                # print(keyframes)
                # print(keyframe_indices)
                for j in range(1,len(keyframe_indices)-1):
                    # print(j)
                    fid = keyframe_indices[j]
                    kf_type = keyframes[fid]
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
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1
                    # all_fix.append(fixations)
                    # One plot showing both novice and expert numbers for objects, other
                    # start_idx = end_idx

                video_target_acc[user[2:]+'_'+str(seg)] = target_acc
                # video_target_acc.append(target_acc)

    # print(all_fix)
    with open('2b_kt_novice_3trials.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        # kf_names = kt_target_acc[novices[4][2:]].keys()
        kf_names = kt_target_acc[kt_target_acc.keys()[0]].keys()
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in kt_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)

    with open('2b_video_novice_3trials.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = video_target_acc[0].keys()
        # kf_names = video_target_acc[novices[4][2:]].keys()
        kf_names = video_target_acc[video_target_acc.keys()[0]].keys()
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in video_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1  for i in kf_names]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)



if args.eid == '2e':
    print('Percentage accuarcy to predict reference frame per keyframe for second demo')
    print('Measure differences between novice and experts - KT demos')
    # all_expert_fix, all_novice_fix = {}, {}
    target_objects = {
        'Reaching': [['green', 'pasta'], ['yellow', 'pasta']],
        'Grasping': [['green', 'pasta'], ['yellow', 'pasta']],
        'Transport': [['red'], ['blue']],
        'Pouring': [['red', 'pasta'],['blue','pasta']],
        'Return': [['other'],['other']],
        'Release': [['green'],['yellow']]
    }
    expert_target_acc, novice_target_acc = {}, {}

    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[expert_target_acc,novice_target_acc]):

        # Get all expert/novice users
        print("processing users' KT Demos...")
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

                if(int(seg)!=2 and int(seg)!=5):
                    continue

                if demo_type!='k':
                    continue

                bag_file = ''
                if(demo_type=='k'):
                    for file in bagfiles:
                        if (file.endswith("kt-p2.bag")):
                            bag_file = bagloc + file
                    
                    if bag_file == '':
                        print('Bag file does not exist for KT demo, skipping...')
                        continue

                target_acc = {
                    'Reaching': [0, 0],
                    'Grasping': [0, 0],
                    'Transport': [0, 0],
                    'Pouring': [0, 0],
                    'Return': [0, 0],
                    'Release': [0, 0]
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
                            if(val==-1):
                                continue
                            if val>max_val:
                                max_val = val
                                max_color = key

                        if(max_val>0):
                            if max_color == target_objects[kf_type][pouring_round][0]:
                                target_acc[kf_type][0] += 1
                            target_acc[kf_type][1] += 1
                        # all_fix.append(fixations)
                        # One plot showing both novice and expert numbers for objects, other
                        start_idx = end_idx
                    # kt_target_acc.append(target_acc)
                    all_fix[user[2:]] = target_acc

    # print(all_fix)
    with open('2e_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        kf_names = expert_target_acc[experts[0][2:]].keys()
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in expert_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)

    with open('2e_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        kf_names = novice_target_acc[novices[0][2:]].keys()
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in novice_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)


if args.eid == '2f':
    print('Perecentage accuarcy to predict reference frame per keyframe for third demo')
    print('Measure differences between novice and experts - KT demos')
    target_objects = {
        'Reaching': [['green', 'pasta'], ['yellow', 'pasta']],
        'Grasping': [['green', 'pasta'], ['yellow', 'pasta']],
        'Transport': [['red'], ['blue']],
        'Pouring': [['red', 'pasta'],['blue','pasta']],
        'Return': [['other'],['other']],
        'Release': [['green'],['yellow']]
    }
    expert_target_acc, novice_target_acc = {}, {}

    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[expert_target_acc,novice_target_acc]):

        # Get all expert/novice users
        print("processing users' KT Demos...")
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

                if(int(seg)!=3 and int(seg)!=6):
                    continue

                if demo_type!='k':
                    continue

                bag_file = ''
                if(demo_type=='k'):
                    for file in bagfiles:
                        if (file.endswith("kt-p3.bag")):
                            bag_file = bagloc + file
                    
                    if bag_file == '':
                        print('Bag file does not exist for KT demo, skipping...')
                        continue

                target_acc = {
                    'Reaching': [0, 0],
                    'Grasping': [0, 0],
                    'Transport': [0, 0],
                    'Pouring': [0, 0],
                    'Return': [0, 0],
                    'Release': [0, 0]
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
                            if(val==-1):
                                continue
                            if val>max_val:
                                max_val = val
                                max_color = key

                        if(max_val>0):
                            if max_color == target_objects[kf_type][pouring_round][0]:
                                target_acc[kf_type][0] += 1
                            target_acc[kf_type][1] += 1
                        # all_fix.append(fixations)
                        # One plot showing both novice and expert numbers for objects, other
                        start_idx = end_idx
                    # kt_target_acc.append(target_acc)
                    all_fix[user[2:]] = target_acc

    # print(all_fix)
    with open('2f_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        kf_names = expert_target_acc[experts[0][2:]].keys()
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in expert_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)

    with open('2f_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        kf_names = novice_target_acc[novices[0][2:]].keys()
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in novice_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)



if args.eid == '2h':
    print('Percentage of time during entire demo - spent on objects, gripper or other parts of workspace for third demo')
    print('Measure differences between novice and experts - KT demos')
    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert/novice users
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

                if(int(seg)!=3 and int(seg)!=6):
                    continue

                if demo_type!='k':
                    continue

                bag_file = ''
                for file in bagfiles:
                    if (file.endswith("kt-p3.bag")):
                        bag_file = bagloc + file
                
                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
                # print(keyframes)
                start_idx, end_idx = keyframes[0], keyframes[-1]
                hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)

                fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx,end_idx)
                # all_fix.append(fixations)
                all_fix[user[2:]] = fixations
                # One plot showing both novice and expert numbers for objects, other

    # print(all_fix)
    with open('2h_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[0].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('2h_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[0].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)



if args.eid == '3a':
    print('Do long fixations line up with keyframes?')
    print('Video versus KT demos (expert users)')

    plt.figure(1, figsize=(20,5))
    plt.figure(2, figsize=(20,5))

    # kt_target_acc, video_target_acc = {}, {}

    # target_objects = {
    #     'Reaching': [['green', 'pasta'], ['yellow', 'pasta']],
    #     'Grasping': [['green', 'pasta'], ['yellow', 'pasta']],
    #     'Transport': [['red'], ['blue']],
    #     'Pouring': [['red', 'pasta'],['blue','pasta']],
    #     'Return': [['other'],['other']],
    #     'Release': [['green'],['yellow']]
    # }
    user_dir = expert_dir
    print("processing Expert Users' Video Demos...")
    # for i in range(len(experts)):
    for i in range(1):
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

            # if demo_type!='k':
            #     continue

            bag_file = ''
            if(demo_type=='k'):
                for file in bagfiles:
                    if (file.endswith("kt-p1.bag")):
                        bag_file = bagloc + file
                
                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

            # target_acc = {
            #     'Reaching': [0, 0],
            #     'Grasping': [0, 0],
            #     'Transport': [0, 0],
            #     'Pouring': [0, 0],
            #     'Return': [0, 0],
            #     'Release': [0, 0]
            # }

            keyframe_color = {
                'Reaching': 'navy',
                'Grasping': 'orange',
                'Close': 'purple',
                'Transport': 'peru',
                'Pouring': 'k',
                'Return': 'salmon',
                'Open': 'darkolivegreen',
                'Release': 'lightskyblue',
                'Other': 'grey'
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
                end_idx = keyframe_indices[-1] 
                fixation_color_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

                plt.figure(1)
                plt.scatter(fixation_idx_list,np.repeat(i,len(fixation_idx_list)),color=fixation_color_list, s=5, marker='^') #, marker='|'

                # Mark keyframe boundaries
                for fid in keyframe_indices:
                    kf_type = keyframes[fid]
                    plt.vlines(x=fid, color=keyframe_color[kf_type], linestyle='--', ymin=i-0.3, ymax=i+0.3, label=kf_type)

                # Mark keyframe range label
                for j in range(1,len(keyframe_indices)):
                    print(keyframe_indices[j-1], keyframe_indices[j])
                    # plt.axhline(y=i+0.2, xmin=keyframe_indices[j-1], xmax=keyframe_indices[j], linewidth = 4, color = 'black')
                    plt.hlines(y=i+0.2, xmin=keyframe_indices[j-1]+10, xmax=keyframe_indices[j]-10, colors='k')

            if(demo_type=='v'):
                start_idx = keyframe_indices[0] 
                end_idx = keyframe_indices[-2] 
                fixation_color_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

                plt.figure(2)
                plt.scatter(fixation_idx_list,np.repeat(i,len(fixation_idx_list)),color=fixation_color_list, s=5, marker='^') #, marker='|'

                # Mark keyframe boundaries
                for fid in keyframe_indices[1:-2]:
                    kf_type = keyframes[fid]
                    plt.vlines(x=fid, color=keyframe_color[kf_type], linestyle='--', ymin=i-0.3, ymax=i+0.3, label=kf_type)

                # Mark keyframe range label
                for j in range(0,len(keyframe_indices)-1):
                    # plt.axhline(y=i+0.2, xmin=keyframe_indices[j]-2, xmax=keyframe_indices[j+1]+2, linewidth = 4, color = 'black')
                    plt.hlines(y=i+0.2, xmin=keyframe_indices[j]+2, xmax=keyframe_indices[j+1]-2, colors='k')


    plt.figure(1)
    title = 'Pouring Task - Expert users, KT Demos'
    plt.title(title)
    # unique keyframe legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('vis/'+title)


    plt.figure(2)
    title = 'Pouring Task - Expert users, Video Demos'
    plt.title(title)
    # unique keyframe legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('vis/'+title)


            # if(demo_type=='v'):
            #     for j in range(1,len(keyframe_indices)-1):
            #         fid = keyframe_indices[j]
            #         kf_type = keyframes[fid]
            #         if(kf_type=='Pouring'):
            #             first_grasp = True
            #         if kf_type=='Reaching' and first_grasp:
            #             pouring_round = 1
            #         start_idx = fid
            #         end_idx = keyframe_indices[j+1]
            #         fixation_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

            #         # assign max value to the color of the default target of this KF
            #         max_val = 0
            #         for o in target_objects[kf_type][pouring_round]:
            #             if(fixations[o]!=-1):
            #                 max_val += fixations[o]
            #         max_color = target_objects[kf_type][pouring_round][0]
            #         for key, val in fixations.items():
            #             if(val!=-1):
            #                 continue
            #             if val>max_val:
            #                 max_val = val
            #                 max_color = key
            #         if(max_val>0):
            #             if max_color == target_objects[kf_type][pouring_round][0]:
            #                 target_acc[kf_type][0] += 1
            #             target_acc[kf_type][1] += 1
            #     video_target_acc[user[2:]] = target_acc


    # what percentage of long fixations line up with keyframes?
    # with open('3a_kt_expert.csv', mode='w') as expert_file:
    #     expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     kf_names = kt_target_acc[experts[0][2:]].keys()
    #     u_kf_names = ['User ID'] + kf_names
    #     expert_writer.writerow(u_kf_names)
        
    #     # no_of_colors = length(color_names)
    #     for u,acc in kt_target_acc.items():
    #         value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1  for i in kf_names]
    #         value_list = [u] + value_list
    #         expert_writer.writerow(value_list)

    # with open('3a_video_expert.csv', mode='w') as expert_file:
    #     expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     kf_names = video_target_acc[experts[0][2:]].keys()
    #     u_kf_names = ['User ID'] + kf_names
    #     expert_writer.writerow(u_kf_names)
    #     for u,acc in video_target_acc.items():
    #         value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
    #         value_list = [u] + value_list
    #         expert_writer.writerow(value_list)



if args.eid == '3b':
    print('Do long fixations line up with keyframes?')
    print('Video versus KT demos (novice users)')

    plt.figure(1, figsize=(20,5))
    plt.figure(2, figsize=(20,5))

    user_dir = novice_dir
    print("processing Expert Users' Video Demos...")
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

            # target_acc = {
            #     'Reaching': [0, 0],
            #     'Grasping': [0, 0],
            #     'Transport': [0, 0],
            #     'Pouring': [0, 0],
            #     'Return': [0, 0],
            #     'Release': [0, 0]
            # }

            keyframe_color = {
                'Reaching': 'navy',
                'Grasping': 'orange',
                'Close': 'purple',
                'Transport': 'peru',
                'Pouring': 'k',
                'Return': 'salmon',
                'Open': 'darkolivegreen',
                'Release': 'lightskyblue',
                'Other': 'grey'
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

            hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)

            if(demo_type=='k'):
                start_idx = 0  
                end_idx = keyframe_indices[-1] 
                fixation_color_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

                plt.figure(1)
                plt.scatter(fixation_idx_list,np.repeat(i,len(fixation_idx_list)),color=fixation_color_list, s=5, marker='^') #, marker='|'

                # Mark keyframe boundaries
                for fid in keyframe_indices:
                    kf_type = keyframes[fid]
                    plt.vlines(x=fid, color=keyframe_color[kf_type], linestyle='--', ymin=i-0.3, ymax=i+0.3, label=kf_type)

            if(demo_type=='v'):
                start_idx = keyframe_indices[0] 
                end_idx = keyframe_indices[-2] 
                fixation_color_list, fixation_idx_list = filter_fixations_with_timeline(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

                plt.figure(2)
                plt.scatter(fixation_idx_list,np.repeat(i,len(fixation_idx_list)),color=fixation_color_list, s=5, marker='^') #, marker='|'

                # Mark keyframe boundaries
                for fid in keyframe_indices[1:-2]:
                    kf_type = keyframes[fid]
                    plt.vlines(x=fid, color=keyframe_color[kf_type], linestyle='--', ymin=i-0.3, ymax=i+0.3, label=kf_type)


    plt.figure(1)
    title = 'Pouring Task - Novice users, KT Demos'
    plt.title(title)
    # unique keyframe legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('vis/'+title)


    plt.figure(2)
    title = 'Pouring Task - Novice users, Video Demos'
    plt.title(title)
    # unique keyframe legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('vis/'+title)


if args.eid == '4a':
    print('Ignoring the black gripper pixels of fixation and focusing on task objects only')
    print('Measure differences between novice and experts - video demos on task objects only')
    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Expert Users' Video Demos...")
        for i in range(len(u)):
        # for i in [7]:
            user = u[i]
            print(user) #KT1,KT2
            dir_name = os.listdir(user_dir+user)

            a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
            d = os.listdir(a)

            exps = order[user]

            # bagloc = bag_dir + user + '/bags/'
            # bagfiles = os.listdir(bagloc)


            for seg in d:
                print('Segment ', seg)
                demo_type = exps[0] if int(seg)<=3 else exps[1]

                if(int(seg)!=1 and int(seg)!=4):
                    continue

                if demo_type!='v':
                    continue

                # bag_file = ''
                # for file in bagfiles:
                #     if (file.endswith("kt-p1.bag")):
                #         bag_file = bagloc + file
                
                # if bag_file == '':
                #     print('Bag file does not exist for KT demo, skipping...')
                #     continue

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                # keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
                keyframes, all_keyframe_indices = get_video_keyframe_labels(user, video_file, video_kf_file)
                # print keyframes
                if(keyframes==[]):
                    continue
                # start_idx, end_idx = all_keyframe_indices[0], all_keyframe_indices[-1]
                hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)

                fixations = filter_fixations_ignore_black(video_file, model, gp, all_vts, demo_type, saccade_indices, all_keyframe_indices, keyframes)
                # all_fix.append(fixations)
                all_fix[user[2:]] = fixations
                # One plot showing both novice and expert numbers for objects, other

    # print(all_fix)
    with open('4a_video_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[experts[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('4a_video_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[novices[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)


if args.eid == '4b':
    print('Ignoring the black gripper pixels of fixation and focusing on task objects only')
    print('Measure differences between novice and experts - KT demos')
    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Expert Users' KT Demos...")
        for i in range(len(u)):
        # for i in range(1):
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

                bag_file = ''
                for file in bagfiles:
                    if (file.endswith("kt-p1.bag")):
                        bag_file = bagloc + file
                
                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                # keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
                keyframes, all_keyframe_indices = get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
                # print keyframes
                if(keyframes==[]):
                    continue
                # start_idx, end_idx = all_keyframe_indices[0], all_keyframe_indices[-1]
                hsv_timeline, saccade_indices, _ = get_hsv_color_timeline(data, video_file)

                fixations = filter_fixations_ignore_black(video_file, model, gp, all_vts, demo_type, saccade_indices, all_keyframe_indices, keyframes)
                # all_fix.append(fixations)
                all_fix[user[2:]] = fixations
                # One plot showing both novice and expert numbers for objects, other

    # print(all_fix)
    with open('4b_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[experts[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('4b_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[novices[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)






if args.eid == '5a':
    print('Major reference frame before and after a keyframe')
    print('Measure differences between novice and experts - KT demos')

    expert_target_acc, novice_target_acc = {}, {}
    

    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[expert_target_acc,novice_target_acc]):

        # Get all expert/novice users
        print("processing users' KT Demos...")
        for i in range(len(u)):
        # print(u)
        # for i in range(1):
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
                # print(demo_type)

                # if(int(seg)!=1 and int(seg)!=4):
                #     continue

                if demo_type!='k':
                    continue

                bag_file = ''
                if(demo_type=='k'):
                    for file in bagfiles:
                        # if (file.endswith("kt-p1.bag")):
                        #     bag_file = bagloc + file
                        if (file.endswith("kt-p1.bag") and (int(seg)==1 or int(seg)==4)):
                            bag_file = bagloc + file
                        elif (file.endswith("kt-p2.bag") and (int(seg)==2 or int(seg)==5)):
                            bag_file = bagloc + file
                        elif (file.endswith("kt-p3.bag") and (int(seg)==3 or int(seg)==6)):
                            bag_file = bagloc + file
                    
                    if bag_file == '':
                        print('Bag file does not exist for KT demo, skipping...')
                        continue

                target_acc = {
                    'step': [0, 0],
                    'non-step': [0, 0]
                }

             


                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'

                keyframes, keyframe_indices = get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
                if(keyframe_indices==[]):
                    continue
                step_kf_indices = get_step_kf_indices(keyframes, keyframe_indices)
                

                hsv_timeline, saccade_indices, fps = get_hsv_color_timeline(data, video_file)


                for fid in keyframe_indices:

                    start_idx = fid - math.floor(fps)
                    if start_idx<0: 
                        start_idx = 0
                    end_idx = fid + math.floor(fps)
                    if end_idx > len(hsv_timeline):
                        end_idx = len(hsv_timeline) - 1
                    fixations_before = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, fid)
                    fixations_after = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, fid, end_idx)


                    if fid in step_kf_indices:
                        kf_type = 'step'
                    else:
                        kf_type = 'non-step'

                    # assign max value to the color of the default target of this KF
                    max_val_before = 0
                    max_val_after = 0
                    # TODO: major color of a keyframe could involve 2 hsv ranges

                    for key, val in fixations_before.items():
                        if(val==1):
                            print('continuing')
                            continue
                        if val>max_val_before:
                            max_val_before = val
                            max_color_before = key


                    for key, val in fixations_after.items():
                        if(val==-1):
                            continue
                        if val>max_val_after:
                            max_val_after = val
                            max_color_after = key

                    if(max_val_before>0 and max_val_after>0):
                        print(kf_type, keyframes[fid])
                        print('****max colors:')
                        print(max_color_before, max_color_after)
                        if max_color_after!=max_color_before and max_color_before!='other' and max_color_after!='other':
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1

                all_fix[user[2:]] = target_acc
                print(target_acc)

    # print(all_fix)
    with open('5a_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        kf_names_ = expert_target_acc[experts[0][2:]].keys()
        kf_names = [kf_names_[0], kf_names_[0]+ ' total', kf_names_[1], kf_names_[1]+ ' total']
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in expert_target_acc.items():
            # value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [acc[i][j] for i in kf_names_ for j in [0,1]]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)

    with open('5a_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # kf_names = kt_target_acc[0].keys()
        kf_names_ = novice_target_acc[novices[0][2:]].keys()
        kf_names = [kf_names_[0], kf_names_[0]+ ' total', kf_names_[1], kf_names_[1]+ ' total']
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        # no_of_colors = length(color_names)
        for u, acc in novice_target_acc.items():
            # value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [acc[i][j] for i in kf_names_ for j in [0,1]]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)   





if args.eid == '5b':
    print('Major reference frame before and after a keyframe')
    print('Measure importance of gaze as a feature with an ROC curve - KT demos')
    # all_expert_fix, all_novice_fix = {}, {}
    # target_objects = {
    #     'Reaching': [['green', 'pasta'], ['yellow', 'pasta']],
    #     'Grasping': [['green', 'pasta'], ['yellow', 'pasta']],
    #     'Transport': [['red'], ['blue']],
    #     'Pouring': [['red', 'pasta'],['blue','pasta']],
    #     'Return': [['other'],['other']],
    #     'Release': [['green'],['yellow']]
    # }
    expert_target_acc, novice_target_acc = {}, {}

    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[expert_target_acc,novice_target_acc]):

        # Get all expert/novice users
        print("processing users' KT Demos...")
        for i in range(len(u)):
        # print(u)
        # for i in range(1):
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
                # print(demo_type)

                if(int(seg)!=1 and int(seg)!=4):
                    continue

                if demo_type!='k':
                    continue

                bag_file = ''
                if(demo_type=='k'):
                    for file in bagfiles:
                        if (file.endswith("kt-p1.bag")):
                            bag_file = bagloc + file
                    
                    if bag_file == '':
                        print('Bag file does not exist for KT demo, skipping...')
                        continue

                target_count = {
                    'step': [],         # a list of target frame counts before and after per keyframe
                    'non-step': []
                }

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'

                keyframes, keyframe_indices = get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
                if(keyframe_indices==[]):
                    continue
                step_kf_indices = get_step_kf_indices(keyframes, keyframe_indices)
                

                hsv_timeline, saccade_indices, fps = get_hsv_color_timeline(data, video_file)


                for fid in keyframe_indices:

                    start_idx = fid - 3*math.floor(fps)
                    if start_idx<0: 
                        start_idx = 0
                    end_idx = fid + 3*math.floor(fps)
                    if end_idx > len(hsv_timeline):
                        end_idx = len(hsv_timeline) - 1
                    fixations_before = filter_fixation_counts(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, fid)
                    fixations_after = filter_fixation_counts(video_file, model, gp, all_vts, demo_type, saccade_indices, fid, end_idx)

                    colored_frames_before = 0
                    colored_frames_after = 0
                    for key, val in fixations_before.items():
                        if key=='red' or key=='blue' or key=='green' or key=='yellow' or key=='pasta':
                            colored_frames_before+=val

                    for key, val in fixations_after.items():
                        if key=='red' or key=='blue' or key=='green' or key=='yellow' or key=='pasta':
                            colored_frames_after+=val

                    if fid in step_kf_indices:
                        kf_type = 'step'
                    else:
                        kf_type = 'non-step'

                    # assign max value to the color of the default target of this KF
                    max_val_before = 0
                    max_val_after = 0
                    # TODO: major color of a keyframe could involve 2 hsv ranges

                    for key, val in fixations_before.items():
                        if(val==-1):
                            print('continuing')
                            continue
                        if colored_frames_before>0:
                            if val*100.0/colored_frames_before>max_val_before:
                                max_val_before = val*100.0/colored_frames_before
                                max_color_before = key


                    for key, val in fixations_after.items():
                        if(val==-1):
                            continue
                        if colored_frames_after>0:
                            if val*100.0/colored_frames_after>max_val_after:
                                # print colored_frames_after
                                max_val_after = val*100.0/colored_frames_after
                                max_color_after = key

                    if(max_val_before>0 and max_val_after>0):
                        val_color_after = fixations_after[max_color_before]
                        #if max_color_after!=max_color_before and max_color_before!='other':# and max_color_after!='other':
                        target_count[kf_type].append(abs(max_val_before-val_color_after))
                        #else:
                        #    target_count[kf_type].append(0)

                all_fix[user[2:]] = target_count
                print(target_count)

    # TPR, FPR = {}, {}
    thresholds = range(0,105,5)



    # CSV files for ROC curve
    with open('5b_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['Threshold', 'True Positive Rate', 'False Positive Rate']
        expert_writer.writerow(header)
        # no_of_colors = length(color_names)
        for t in thresholds:
            tp_num, tp_den, fp_num, fp_den  = 0, 0, 0, 0
            for u, diff in expert_target_acc.items():
                step_correct = float(sum(i > t for i in diff['step']))
                # step_incorrect = len(diff['step']) - step_correct
                non_step_correct = float(sum(i < t for i in diff['non-step']))
                # non_step_incorrect = len(diff['non-step']) - non_step_correct
                tp_num += step_correct
                tp_den += len(diff['step'])
                fp_num += non_step_correct
                fp_den += len(diff['non-step'])
            tp = tp_num/tp_den
            fp = fp_num/fp_den
            expert_writer.writerow([t, tp, fp])

    with open('5b_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['Threshold', 'True Positive Rate', 'False Positive Rate']
        novice_writer.writerow(header)
        # no_of_colors = length(color_names)
        for t in thresholds:
            tp_num, tp_den, fp_num, fp_den  = 0, 0, 0, 0
            for u, diff in novice_target_acc.items():
                step_correct = float(sum(i > t for i in diff['step']))
                # step_incorrect = len(diff['step']) - step_correct
                non_step_correct = float(sum(i < t for i in diff['non-step']))
                # non_step_incorrect = len(diff['non-step']) - non_step_correct
                tp_num += step_correct
                tp_den += len(diff['step'])
                fp_num += non_step_correct
                fp_den += len(diff['non-step'])
            tp = tp_num/tp_den
            fp = fp_num/fp_den
            novice_writer.writerow([t, tp, fp])