import cv2
import ast 
import os
import gzip
from sync_hist import get_color_timeline_with_seg
from sync_hist import get_color_timeline
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

keep_saccades = True

bag_dir = '/home/asaran/gaze_lfd_ws/src/hlpr_kinesthetic_teaching/record_demonstration_with_eye_tracker/data/bags/'

# experts - user 1, 2, 4, 5, 6, 8, 12, 13, 18, 19 
# novices - user 3, 7, 9, 10, 11, 14, 15, 16, 17, 20
main_dir = '/home/asaran/gaze_lfd_ws/src/hlpr_kinesthetic_teaching/record_demonstration_with_eye_tracker/data/pouring/experts/'    #IRL
users = os.listdir(main_dir)
print(users)

order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

hist_k, hist_v = {}, {}
hists = {
    'k': hist_k,
    'v': hist_v
}

condition_names = {
    'k': 'KT demo',
    'v': 'Video demo'
}

keyframe_color = {
    'Reaching': 'navy',
    'Grasping': 'orange',
    'Close': 'purple',
    'Transport': 'peru',
    'Pouring': 'k',
    'Return': 'salmon',
    'Open': 'darkolivegreen',
    'Release': 'lightskyblue'
}

plt.figure(1, figsize=(20,5))
plt.figure(2, figsize=(20,5))

# for i in range(0,1):
for i in range(len(users)):
    user = users[i]
    print(user) #KT1,KT2
    dir_name = os.listdir(main_dir+user)

    a = main_dir+user+'/'+dir_name[0]+'/'+'segments/'
    d = os.listdir(a)

    #IRL segment
    if(len(d))>6:
        print('More than 6 segments!! Clean up the data...')
        exit()

    exps = order[user]

    bagloc = bag_dir + user + '/bags/'
    bagfiles = os.listdir(bagloc)

    for seg in d:
        print('Segment ', seg)
        demo_type = exps[0] if int(seg)<=3 else exps[1]

        if(int(seg)!=1 and int(seg)!=4):
            continue

        data = []
        files = os.listdir(a+seg)
        
        for file in files:
            if (file.endswith("json.gz")):
                with gzip.open(a+seg+'/'+file, "rb") as f:
                    data=f.readlines()
                
                    for r in range(len(data)):
                        row = data[r]
                        data[r] = ast.literal_eval(row.strip('\n'))


        video_file = a+seg+'/fullstream.mp4'
        bag_file = ''
        if(demo_type=='k'):
            for file in bagfiles:
                if (file.endswith("kt-p1.bag")):
                    bag_file = bagloc + file
            
            if bag_file == '':
                print('Bag file does not exist for KT demo, skipping...')
                continue
            timeline, keyframes, saccade_indices = get_color_timeline_with_seg(data, video_file, bag_file, keep_saccades)
            # print(user, i, exp_id, bag_file, keyframes, open_keyframe)

        if(demo_type=='v'):
            timeline, saccade_indices = get_color_timeline(data, video_file, keep_saccades)

        # remove saccades
        scale = 10
        scaled_timeline = range(0,len(timeline)*scale,scale)

        if(not keep_saccades):
            print('Removing saccades')
            for index in sorted(saccade_indices, reverse=True):
                del scaled_timeline[index]
                del timeline[index]
                # print('deleted')

        if(demo_type=='k'):    
            plt.figure(1)
            plt.scatter(scaled_timeline,np.repeat(i,len(scaled_timeline)),color=timeline, s=2) #, marker='|'

            # Mark keyframe boundaries
            for k,v in iter(keyframes.items()):
                for xc in v:
                    plt.vlines(x=xc*scale, color=keyframe_color[k], linestyle='--', ymin=i-0.3, ymax=i+0.3, label=k)


                #plt.show()
        if(demo_type=='v'):
            plt.figure(2)
            plt.scatter(scaled_timeline,np.repeat(i,len(scaled_timeline)),color=timeline, s=2) #, marker='|'

            # Mark keyframe boundaries
            # for xc in keyframes:
            #     plt.vlines(x=xc*scale, color='red', linestyle='--', ymin=i-0.3, ymax=i+0.3)

            # for xc in open_keyframe:
            #     plt.vlines(x=xc*scale, color='green', linestyle='--', ymin=i-0.3, ymax=i+0.3)
            #plt.show()

        
plt.figure(1)
title = 'Pouring Task: Expert users, KT Demos'
plt.title(title)
# unique keyframe legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig('vis/'+title)

plt.figure(2)
title = 'Pouring Task: Expert users, Video Demos'
plt.title(title)    
plt.savefig('vis/'+title)