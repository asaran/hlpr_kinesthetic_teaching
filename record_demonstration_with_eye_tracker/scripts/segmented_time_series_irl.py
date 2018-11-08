import cv2
import ast 
import os
import gzip
from sync_hist import get_color_timeline_with_seg
import matplotlib.pyplot as plt
import numpy as np

bag_dir = '/media/akanksha/pearl_Gemini/gaze_lfd_user_study/'

main_dir = '/media/akanksha/pearl_Gemini/IRL/'
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
for i in range(5,6):
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

    # experts - user 1, 2, 4, 5, 6, 8, 12, 13, 18, 19 
    # novices - user 3, 7, 9, 10, 11, 14, 15, 16, 17, 20

    bagloc = bag_dir + user + '/bags/'
    bagfiles = os.listdir(bagloc)
    #print(bagfiles)
    
    # conditions
    for seg in d:
        bagfile = ''
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

        for file in bagfiles:
            if (file.endswith("kt-irl-bowl.bag") and demo_type=='k' and cond=='b'):
                bag_file = bagloc + file
            elif(file.endswith("kt-irl-plate.bag") and demo_type=='k' and cond=='p'):
                bag_file = bagloc + file

        #if(file.endswith(".mp4")):
        if demo_type=='k':
            video_file = a+seg+'/fullstream.mp4'
            print(video_file)
            timeline, keyframes, open_keyframes = get_color_timeline_with_seg(data, video_file, bag_file)
            #fig = plt.figure()
            #for i,c in enumerate(timeline):
            #    print(c[2],c[1],c[0])
            #print(range(0,len(timeline)*10,10))
            plt.scatter(range(0,len(timeline)*10,10),np.repeat(1,len(timeline)),color=timeline, s=5)
            title = 'User #'+str(i)+': '+condition_names[exp_id[0]]+', '+condition_names[exp_id[1]]
            plt.title(title)

            for xc in keyframes:
                plt.axvline(x=xc, color='red', linestyle='--')

            for xc in open_keyframe:
                plt.axvline(x=xc, color='yellow', linestyle='--')

            #plt.show()
            #print(hist)