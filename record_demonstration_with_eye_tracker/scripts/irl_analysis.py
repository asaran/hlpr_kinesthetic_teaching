import cv2
import ast 
import os
import gzip
from sync import sync_func 

main_dir = '/media/akanksha/pearl_Gemini/IRL/'
users = os.listdir(main_dir)
print(users)

order = {'KT1':'kvpb','KT2':'kvbp','KT3':'vkpb','KT4':'vkbp','KT5':'kvbp','KT6':'kvbp','KT7':'vkbp','KT8':'vkpb','KT9':'kvpb','KT10':'kvbp',\
        'KT11':'vkpb','KT12':'vkbp','KT13':'kvbp','KT14':'kvpb','KT15':'vkbp','KT16':'vkpb','KT17':'kvpb','KT18':'vkbp','KT19':'vkpb','KT20':'vkbp'}

for user in users:
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


    for seg in d:
        print('Segment ', seg)
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
        fixations = sync_func(data, video_file)
        print(fixations)

        """
        vidcap = cv2.VideoCapture(a+seg+'/fullstream.mp4')
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        #print fps   #25 fps
        success,image = vidcap.read()

        count = 0
        imgs = []       # list of image frames
        frame2ts = []   # corresponding list of video time stamp values in microseconds
        success = True
        while success:
            #cv2.imwrite("data/imgs/%d.jpg" % count, image)     # save frame as JPEG file   
            #imgs.append(image)   
            success,image = vidcap.read()
            frame2ts.append(int((count/fps)*1000000))
            #print('Read a new frame: ', success)
            count += 1
            #print(count)
        print('read', str(count),'image frames')
        """