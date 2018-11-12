import matplotlib.pyplot as plt
import numpy as np
import ast 
import os
import gzip
from utils import get_cumulative_gaze_dist

# iterate through users for placement task
# find the cumulative distance traveled by gaze points over frame number on the x-axis
# plot cumulative distance for each user on the same plot

main_dir = '../../data/pouring/all_users/'    #IRL
users = os.listdir(main_dir)
print(users)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', \
          'darkgreen', 'plum', 'grey', 'skyblue', 'khaki', 'darkblue', 'teal', 'orange', \
          'lightpink', 'sandybrown']

order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

hist_k, hist_v = {}, {}
hists = {
    'k': hist_k,
    'v': hist_v
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
    if(len(d))>6:
        print('More than 6 segments!! Clean up the data...')
        exit()

    exps = order[user]

    for seg in d:
        print('Segment ', seg)
        demo_type = exps[0] if int(seg)<=3 else exps[1]

        # if(int(seg)!=1 and int(seg)!=4):
        #     continue

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