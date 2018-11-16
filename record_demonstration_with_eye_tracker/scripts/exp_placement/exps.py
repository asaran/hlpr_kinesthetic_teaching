# Experiments for the Pouring Task
# 1. Percentage of time during entire demo - spent on objects, gripper or other parts of workspace 
#       a. Measure differences between novice and experts - video demos
#       b. Measure differences between novice and experts - KT demos
# 2. Perecentage accuarcy to predict reference frame per keyframe (same consecutive keyframes clubbed together) -- NO ANOVA
#       a. Video versus KT demos (expert users)
#       b. Video versus KT demos (novice users)     
#       c. Video demos over time (expert users)
#       d. Video demos over time (novice users)
#       e. KT demos over time (expert users)
#       f. KT demos over time (novice users)


import argparse
from utils import takeClosest, get_hsv_color_timeline, get_color_name_from_hist
from utils import get_video_keyframes, read_json, filter_fixations, get_kt_keyframes
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-eid", type=str, default="1a", help='Experiment ID')
args = parser.parse_args()

expert_dir = '../../data/reward/experts/'    
experts = os.listdir(expert_dir)
# print(users)

novice_dir = '../../data/reward/novices/'    
novices = os.listdir(novice_dir)

all_dir = '../../data/reward/all_users/'
all_users = os.listdir(all_dir)

order = {'KT1':'kvpb','KT2':'kvbp','KT3':'vkpb','KT4':'vkbp','KT5':'kvbp','KT6':'kvpb','KT7':'vkbp','KT8':'vkpb','KT9':'kvpb','KT10':'kvbp',\
        'KT11':'vkpb','KT12':'vkbp','KT13':'kvbp','KT14':'kvpb','KT15':'vkbp','KT16':'vkpb','KT17':'kvpb','KT18':'vkbp','KT19':'vkpb','KT20':'vkbp'}


condition_names = {
    'k': 'KT demo',
    'v': 'Video demo',
    'p': 'plate target',
    'b': 'bowl target'
}

video_kf_file = 'video_kf.txt'
bag_dir = '/home/akanksha/Documents/gaze_for_lfd_study_data/gaze_lfd_user_study/'

if args.eid == '1a':
    print('Percentage of time during entire demo - spent on objects or other parts of workspace')
    print('Measure differences between novice and experts - video demos')

    all_expert_fix, all_novice_fix = [], []
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Expert Users' Video Demos...")
        for i in range(len(u)):
            user = u[i]
            print(user) #KT1,KT2
            if user== 'KT13':
                continue
            dir_name = os.listdir(user_dir+user)

            a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
            d = os.listdir(a)

            exps = order[user]

            for seg in d:
                print('Segment ', seg)
                demo_type = exps[0] if int(seg)<=2 else exps[1]
                cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
                exp_id = demo_type + cond

                if demo_type!='v':
                    continue

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)
                keyframe_indices = get_video_keyframes(user, int(seg), video_file, video_kf_file)
                print(keyframe_indices)
                start_idx, end_idx = keyframe_indices['Start'][0], keyframe_indices['Stop'][0]
                fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)
                all_fix.append(fixations)
            # One plot showing both novice and expert numbers for objects, other

    # print(all_expert_fix)
    # print(all_novice_fix)
    with open('1a_video_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[0].keys()
        expert_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for expert_fix in all_expert_fix:
            value_list = [expert_fix[i] for i in color_names]
            expert_writer.writerow(value_list)

    with open('1a_video_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[0].keys()
        novice_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for novice_fix in all_novice_fix:
            value_list = [novice_fix[i] for i in color_names]
            novice_writer.writerow(value_list)

if args.eid == '1b':
    print('Percentage of time during entire demo - spent on objects, gripper or other parts of workspace')
    print('Measure differences between novice and experts - KT demos')
    all_expert_fix, all_novice_fix = [], []
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
                demo_type = exps[0] if int(seg)<=2 else exps[1]
                cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
                exp_id = demo_type + cond


                if demo_type!='k':
                    continue

                bag_file = ''
                for file in bagfiles:
                    if (file.endswith("kt-irl-bowl.bag") and demo_type=='k' and cond=='b'):
                        bag_file = bagloc + file
                    elif(file.endswith("kt-irl-plate.bag") and demo_type=='k' and cond=='p'):
                        bag_file = bagloc + file
                
                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

                data, gp, model, all_vts = read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
                print(keyframes)
                start_idx, end_idx = keyframes[0], keyframes[-1]
                hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)

                fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx,end_idx)
                all_fix.append(fixations)
                # One plot showing both novice and expert numbers for objects, other

    # print(all_fix)
    with open('1b_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[0].keys()
        expert_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for expert_fix in all_expert_fix:
            value_list = [expert_fix[i] for i in color_names]
            expert_writer.writerow(value_list)

    with open('1b_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[0].keys()
        novice_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for novice_fix in all_novice_fix:
            value_list = [novice_fix[i] for i in color_names]
            novice_writer.writerow(value_list)

if args.eid == '2a':
    print('Perecentage accuarcy to predict instruction from gaze')
    print('Plate versus Bowl (Video demo for all users)')

    user_dir = all_dir
    bowl_fixations, plate_fixations = [], []
    for i in range(len(all_users)):
        user = all_users[i]
        print(user) #KT1,KT2
        if user== 'KT13':
            continue
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=2 else exps[1]
            cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
            exp_id = demo_type + cond

            if demo_type!='v':
                continue

            data, gp, model, all_vts = read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)
            keyframe_indices = get_video_keyframes(user, int(seg), video_file, video_kf_file)
            print(keyframe_indices)
            start_idx, end_idx = keyframe_indices['Start'][0], keyframe_indices['Stop'][0]
            fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

            if(cond=='p'):
                plate_fixations.append(fixations)
            if (cond=='b'):
                bowl_fixations.append(fixations)


    with open('2a_video_plate_all.csv', mode='w') as plate_file:
        plate_writer = csv.writer(plate_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = plate_fixations[0].keys()
        plate_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for fix in plate_fixations:
            value_list = [fix[i] for i in color_names]
            plate_writer.writerow(value_list)

    with open('2a_video_bowl_all.csv', mode='w') as bowl_file:
        bowl_writer = csv.writer(bowl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = bowl_fixations[0].keys()
        bowl_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for fix in bowl_fixations:
            value_list = [fix[i] for i in color_names]
            bowl_writer.writerow(value_list)
    

if args.eid == '2b':
    print('Perecentage accuarcy to predict instruction from gaze')
    print('Plate versus Bowl (KT demo for expert users)')
    
    user_dir = expert_dir
    bowl_fixations, plate_fixations = [], []
    for i in range(len(experts)):
        user = experts[i]
        print(user) #KT1,KT2
        # if user== 'KT13':
        #     continue
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)


        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=2 else exps[1]
            cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
            exp_id = demo_type + cond


            if demo_type!='k':
                continue

            bag_file = ''
            for file in bagfiles:
                if (file.endswith("kt-irl-bowl.bag") and demo_type=='k' and cond=='b'):
                    bag_file = bagloc + file
                elif(file.endswith("kt-irl-plate.bag") and demo_type=='k' and cond=='p'):
                    bag_file = bagloc + file
            
            if bag_file == '':
                print('Bag file does not exist for KT demo, skipping...')
                continue

            data, gp, model, all_vts = read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
            print(keyframes)
            start_idx, end_idx = keyframes[0], keyframes[-1]
            hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)
            fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

            if(cond=='p'):
                plate_fixations.append(fixations)
            if (cond=='b'):
                bowl_fixations.append(fixations)


    with open('2b_KT_plate_experts.csv', mode='w') as plate_file:
        plate_writer = csv.writer(plate_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = plate_fixations[0].keys()
        plate_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for fix in plate_fixations:
            value_list = [fix[i] for i in color_names]
            plate_writer.writerow(value_list)

    with open('2b_KT_bowl_experts.csv', mode='w') as bowl_file:
        bowl_writer = csv.writer(bowl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = bowl_fixations[0].keys()
        bowl_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for fix in bowl_fixations:
            value_list = [fix[i] for i in color_names]
            bowl_writer.writerow(value_list)


if args.eid == '2c':
    print('Perecentage accuarcy to predict instruction from gaze')
    print('Plate versus Bowl (KT demo for novice users)')

    user_dir = novice_dir
    bowl_fixations, plate_fixations = [], []
    for i in range(len(novices)):
        user = novices[i]
        print(user) #KT1,KT2
        # if user== 'KT13':
        #     continue
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)


        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=2 else exps[1]
            cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
            exp_id = demo_type + cond


            if demo_type!='k':
                continue

            bag_file = ''
            for file in bagfiles:
                if (file.endswith("kt-irl-bowl.bag") and demo_type=='k' and cond=='b'):
                    bag_file = bagloc + file
                elif(file.endswith("kt-irl-plate.bag") and demo_type=='k' and cond=='p'):
                    bag_file = bagloc + file
            
            if bag_file == '':
                print('Bag file does not exist for KT demo, skipping...')
                continue

            data, gp, model, all_vts = read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
            print(keyframes)
            start_idx, end_idx = keyframes[0], keyframes[-1]
            hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)
            fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

            if(cond=='p'):
                plate_fixations.append(fixations)
            if (cond=='b'):
                bowl_fixations.append(fixations)


    with open('2c_KT_plate_novice.csv', mode='w') as plate_file:
        plate_writer = csv.writer(plate_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = plate_fixations[0].keys()
        plate_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for fix in plate_fixations:
            value_list = [fix[i] for i in color_names]
            plate_writer.writerow(value_list)

    with open('2c_KT_bowl_novice.csv', mode='w') as bowl_file:
        bowl_writer = csv.writer(bowl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = bowl_fixations[0].keys()
        bowl_writer.writerow(color_names)
        # no_of_colors = length(color_names)
        for fix in bowl_fixations:
            value_list = [fix[i] for i in color_names]
            bowl_writer.writerow(value_list)
    

