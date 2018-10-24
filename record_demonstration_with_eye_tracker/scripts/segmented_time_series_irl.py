import cv2
import ast 
import os
import gzip
from sync_hist import get_color_timeline
import matplotlib.pyplot as plt
import numpy as np


main_dir = '/media/asaran/pearl_Gemini/IRL/'
users = os.listdir(main_dir)
print(users)

order = {'KT1':'kvpb','KT2':'kvbp','KT3':'vkpb','KT4':'vkbp','KT5':'kvbp','KT6':'kvbp','KT7':'vkbp','KT8':'vkpb','KT9':'kvpb','KT10':'kvbp',\
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


# naive versus expert users

#for user in users:
for i in range(0,2):
    user = users[i]
    print(user) #KT1,KT2
    dir_name = os.listdir(main_dir+user)
    #print(dir_name)
    a = main_dir+user+'/'+dir_name[0]+'/'+'segments/'
    d = os.listdir(a)
    #print(d) #[1,2,3,4]
    #IRL segment
    if(len(d))!=4:
        print('More than 4 segments!! Clean up the data...')
        exit()

    exps = order[user]

    # KT segments within each condition experiment 
    # read bagfile, get corresponding gaze video timestamp
    # put a marker in the graph for each KT segment recording


    # conditions
    for seg in d:
        print('Segment ', seg)
        #if(int(seg)==3 or int(seg)==4):
        #    continue
        demo_type = exps[0] if int(seg)<=2 else exps[1]
        cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
        exp_id = demo_type + cond

        data = []
        files = os.listdir(a+seg)
        
        for file in files:
            #print d
            if (file.endswith("json.gz")):
                #print('extracted')
                with gzip.open(a+seg+'/'+file, "rb") as f:
                    #print(f)
                    data=f.readlines()
                
                    for r in range(len(data)):
                        row = data[r]
                        data[r] = ast.literal_eval(row.strip('\n'))
        #print(data)

        #if(file.endswith(".mp4")):
        video_file = a+seg+'/fullstream.mp4'
        timeline = get_color_timeline(data, video_file)
        #fig = plt.figure()
        #for i,c in enumerate(timeline):
        #    print(c[2],c[1],c[0])
        #print(range(0,len(timeline)*10,10))
        plt.scatter(range(0,len(timeline)*10,10),np.repeat(1,len(timeline)),color=timeline, s=5)
        title = 'User #'+str(i)+': '+condition_names[exp_id[0]]+', '+condition_names[exp_id[1]]
        plt.title(title)
        plt.show()
        #print(hist)