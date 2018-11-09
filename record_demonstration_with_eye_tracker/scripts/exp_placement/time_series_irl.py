import cv2
import ast 
import os
import gzip
from sync_hist import get_color_timeline_with_seg
import matplotlib.pyplot as plt
import numpy as np

keep_saccades = True

bag_dir = '/home/asaran/gaze_lfd_ws/src/hlpr_kinesthetic_teaching/record_demonstration_with_eye_tracker/data/bags/'

# experts - user 1, 2, 4, 5, 6, 8, 12, 13, 18, 19 
# novices - user 3, 7, 9, 10, 11, 14, 15, 16, 17, 20
main_dir = '/home/asaran/gaze_lfd_ws/src/hlpr_kinesthetic_teaching/record_demonstration_with_eye_tracker/data/reward/novices/'    #IRL
users = os.listdir(main_dir)
print(users)

order = {'KT1':'kvpb','KT2':'kvbp','KT3':'vkpb','KT4':'vkbp','KT5':'kvbp','KT6':'kvpb','KT7':'vkbp','KT8':'vkpb','KT9':'kvpb','KT10':'kvbp',\
        'KT11':'vkpb','KT12':'vkbp','KT13':'kvbp','KT14':'kvpb','KT15':'vkbp','KT16':'vkpb','KT17':'kvpb','KT18':'vkbp','KT19':'vkpb','KT20':'vkbp'}

hist_kp, hist_kb, hist_vp, hist_vb = {}, {}, {}, {}
hists = {
    'kp': hist_kp,
    'kb': hist_kb,
    'vp': hist_vp,
    'vb': hist_vb
}

condition_names = {
    'k': 'KT demo',
    'v': 'Video demo',
    'p': 'plate target',
    'b': 'bowl target'
}


plt.figure(1, figsize=(20,5))
plt.figure(2, figsize=(20,5))

#for user in users:
for i in range(len(users)):
    user = users[i]
    print(user) #KT1,KT2
    dir_name = os.listdir(main_dir+user)

    a = main_dir+user+'/'+dir_name[0]+'/'+'segments/'
    d = os.listdir(a)

    #IRL segment
    if(len(d))>4:
        print('More than 4 segments!! Clean up the data...')
        exit()

    exps = order[user]

    bagloc = bag_dir + user + '/bags/'
    bagfiles = os.listdir(bagloc)

    for seg in d:
        print('Segment ', seg)
        demo_type = exps[0] if int(seg)<=2 else exps[1]
        cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
        exp_id = demo_type + cond

        data = []
        files = os.listdir(a+seg)
        
        for file in files:
            if (file.endswith("json.gz")):
                with gzip.open(a+seg+'/'+file, "rb") as f:
                    data=f.readlines()
                
                    for r in range(len(data)):
                        row = data[r]
                        data[r] = ast.literal_eval(row.strip('\n'))

        for file in bagfiles:
            if (file.endswith("kt-irl-bowl.bag") and demo_type=='k' and cond=='b'):
                bag_file = bagloc + file
            elif(file.endswith("kt-irl-plate.bag") and demo_type=='k' and cond=='p'):
                bag_file = bagloc + file

        if(demo_type=='k'):
            video_file = a+seg+'/fullstream.mp4'
            timeline, keyframes, open_keyframe, saccade_indices = get_color_timeline_with_seg(data, video_file, bag_file, keep_saccades)
            # print(user, i, exp_id, bag_file, keyframes, open_keyframe)

            # remove saccades
            scale = 10
            scaled_timeline = range(0,len(timeline)*scale,scale)

            if(not keep_saccades):
                print('Removing saccades')
                for index in sorted(saccade_indices, reverse=True):
                    del scaled_timeline[index]
                    del timeline[index]
                    # print('deleted')

            if(cond=='p'):
                plt.figure(1)
                plt.scatter(scaled_timeline,np.repeat(i,len(scaled_timeline)),color=timeline, s=5, marker='|')

                # Mark keyframe boundaries
                for xc in keyframes:
                    plt.vlines(x=xc*scale, color='red', linestyle='--', ymin=i-0.3, ymax=i+0.3)

                for xc in open_keyframe:
                    plt.vlines(x=xc*scale, color='green', linestyle='--', ymin=i-0.3, ymax=i+0.3)

                #plt.show()

            if(cond=='b'):
                plt.figure(2)
                plt.scatter(scaled_timeline,np.repeat(i,len(scaled_timeline)),color=timeline, s=5, marker='|')

                # Mark keyframe boundaries
                for xc in keyframes:
                    plt.vlines(x=xc*scale, color='red', linestyle='--', ymin=i-0.3, ymax=i+0.3)

                for xc in open_keyframe:
                    plt.vlines(x=xc*scale, color='green', linestyle='--', ymin=i-0.3, ymax=i+0.3)

                #plt.show()

        


plt.figure(1)
title = 'Naive users KT demo: red plate target'
plt.title(title)    
plt.savefig('vis/'+title)

plt.figure(2)
title = 'Naive users KT demo: yellow bowl target'
plt.title(title)    
plt.savefig('vis/'+title)