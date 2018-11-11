import matplotlib.pyplot as plt
import numpy as np
import ast 
import os
import gzip
from utils import get_cumulative_gaze_dist

# iterate through users for placement task
# find the cumulative distance traveled by gaze points over frame number on the x-axis
# plot cumulative distance for each user on the same plot

main_dir = '/home/asaran/gaze_lfd_ws/src/hlpr_kinesthetic_teaching/record_demonstration_with_eye_tracker/data/reward/all_users/'    #IRL
users = os.listdir(main_dir)
print(users)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', \
          'darkgreen', 'plum', 'grey', 'skyblue', 'khaki', 'darkblue', 'teal', 'orange', \
          'lightpink', 'sandybrown']

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

        video_file = a+seg+'/fullstream.mp4'
        if(os.path.isfile(video_file)):
            cumulative_dist = get_cumulative_gaze_dist(data, video_file)
            print('computed cumulative distance')

        print('plotting cumulative distance ...')
        if(demo_type=='k'):
            plt.figure(1)
            plt.scatter(range(i*500,len(cumulative_dist)+i*500),cumulative_dist,color=colors[i], s=3)

        if(demo_type=='v'):
            plt.figure(2)
            plt.scatter(range(i*500,len(cumulative_dist)+i*500),cumulative_dist,color=colors[i], s=3)


plt.figure(1)
title = 'All users KT demo: cumulative gaze distance'
plt.title(title)    
plt.savefig('vis/'+title)

plt.figure(2)
title = 'All users Video demo: cumulative gaze distance'
plt.title(title)    
plt.savefig('vis/'+title)