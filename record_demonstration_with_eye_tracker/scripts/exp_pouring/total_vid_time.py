import os
from utils import read_json
import cv2

order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

main_dir = '../../data/pouring/all_users/'    #IRL
users = os.listdir(main_dir)
print(users)

video_time = 0
KT_time = 0

# Get the total video data time and KT data
for i in range(len(users)):
    user = users[i]
    print(user) #KT1,KT2
    dir_name = os.listdir(main_dir+user)

    a = main_dir+user+'/'+dir_name[0]+'/'+'segments/'
    d = os.listdir(a)

    exps = order[user]

    for seg in d:
        print('Segment ', seg)
        demo_type = exps[0] if int(seg)<=3 else exps[1]

        if(int(seg)!=1 and int(seg)!=4):
            continue

        data, gp, model, all_vts = read_json(a+seg)
        video_file = a+seg+'/fullstream.mp4'

        vidcap = cv2.VideoCapture(video_file)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if(demo_type=='k'):
            KT_time += length/fps
        if demo_type=='v':
            video_time += length/fps


print('Total video demo time: ', video_time)
print('Total KT demo time: ', KT_time)