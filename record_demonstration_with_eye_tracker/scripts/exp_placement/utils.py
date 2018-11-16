import cv2
import ast
from bisect import bisect_left
from numpy import ones,vstack
from numpy.linalg import lstsq
import numpy as np
import math
import rosbag
import math
import gzip
import os

hist = {
    0: 0,
    10: 0,
    20: 0,
    30: 0,
    40: 0,
    50: 0,
    60: 0,
    70: 0,
    80: 0,
    90: 0,
    100: 0,
    110: 0,
    120: 0,
    130: 0,
    140: 0,
    150: 0,
    160: 0,
    170: 0,
    180: 0,
    190: 0,
    200: 0,
    210: 0,
    220: 0,
    230: 0,
    240: 0,
    250: 0
}


def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], 0
    if pos == len(myList):
        return myList[-1], len(myList)-1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after, pos
    else:
       return before, pos-1

def color_dist(color1, color2):
    r1,g1,b1 = color1
    r2,g2,b2 = color2
    color_d = pow(r1-r2,2) + pow(g1-g2,2) + pow(b1-b2,2)
    mean_rgb = ((r1+r2)/2, (g1+g2)/2, (b1+b2)/2)
    return color_d, mean_rgb

def pixel_dist(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    d = pow(x1-x2,2) + pow(y1-y2,2)
    return math.sqrt(d)

def is_known_color(color):
    known_colors = {
    'red': [[170,190],[65,180],[60,90]],
    'green': [[95,105],[85,115],[65,95]],
    'yellow': [[10,25],[130,160],[110,150]]
    }

    lower_red = np.array([170,65,60])
    upper_red = np.array([190,180,90])

    lower_yellow = np.array([10,130,110])
    upper_yellow = np.array([25,160,150])

    h,s,v = color
    for color in known_colors.keys():
        if h>color[0][0] and h<color[0][1]:
            if s>color[1][0] and s<color[1][1]:
                if v>color[2][0] and v<color[2][1]:
                    return color
    return None



def sync_func(data, video_file):

    fixations = {'red': [], 'yellow':[], 'green':[], 'other':[]}
    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                #start_frame_ts = d['ts']
                #print str(d['vts']) + '\t\t' + str(d['ts']/1000000.)
                vid2ts[d['vts']] = d['ts']
            else:
                #vid_time = d['ts'] - d['vts']
                vid2ts[d['vts']] = d['ts']
                #print str(d['vts']/1000000.)  + '\t\t' + str(d['ts']/1000000.)

        # TODO: if multiple detections for same time stamp?
        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')



    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        #print("Line Solution is ts = {m}vts + {c}".format(m=m,c=c))
        model.append((m,c))
        a = all_vts[i]


    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    #print fps  #25 fps
    success, img = vidcap.read()

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #video = cv2.VideoWriter('gaze.avi',fourcc,fps,(1920,1080))

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds

    while success:          
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #print(count)
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)
        #print('Read a new frame: ', success)
        #print('frame_ts: ', frame_ts)
        #print(all_vts)
        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1
        # print(idx)
        # start = less[-1]
        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        #print('ts: ',ts)
        tracker_ts,_ = takeClosest(all_ts,ts)
        #print('tracker_ts_pos: ', tracker_ts)
        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
        #print(gaze_coords)
        h,s,v = img[gaze_coords[1]][gaze_coords[0]]
        if(count==0):
        #   last_fixation_color = (h,s,v)
            t = 0
            last_gaze_pt = gaze_coords

        p_dist = pixel_dist(gaze_coords,last_gaze_pt)
        if(p_dist<10):
            t = t + int((1.0/fps)*1000000)

        else:
            if(t>100000):
                hist[(h/10)*10] += 1 
            t = 0
        # cd, mean_hsv = color_dist((h,s,v),last_fixation_color)
        # if(cd<10):
        #   t = t + int((1.0/fps)*1000)
            
        # else:
        #   if(t>100):
        #       if is_known_color(mean_hsv) is not None:
        #           fixations[is_known_color(mean_hsv)].append(t)
        #       else:
        #           fixations['other'].append(t)
        #   t = 0

        # #cv2.circle(img,(int(gaze[0]*1920), int(gaze[1]*1080)), 5, (0,255,0), -1)
        # #video.write(img)

        # last_fixation_color = (h,s,v)
        last_gaze_pt = gaze_coords

        count += 1
        success, img = vidcap.read()



    vidcap.release()
    cv2.destroyAllWindows()
    #video.release()

    return hist


# returns a list of rgb color values for gaze point for each video frame
def get_color_timeline(data, video_file):
    timeline = []
    fixations = {'red': [], 'yellow':[], 'green':[], 'other':[]}
    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                #vid_time = d['ts'] - d['vts']
                vid2ts[d['vts']] = d['ts']

        # TODO: if multiple detections for same time stamp?
        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')



    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        model.append((m,c))
        a = all_vts[i]


    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds

    while success:          
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts,  _ = takeClosest(all_ts,ts)
        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
        #h,s,v = img[gaze_coords[1]][gaze_coords[0]]
        b, g, r = img[gaze_coords[1]][gaze_coords[0]]
        instant_color = [r/255.0,g/255.0,b/255.0]
        timeline.append(instant_color)

        last_gaze_pt = gaze_coords

        count += 1
        success, img = vidcap.read()



    vidcap.release()
    cv2.destroyAllWindows()
    return timeline



# returns a rgb color timeline for gaze points along with the frame indices for the recorded keyframes
def get_color_timeline_with_seg(data, video_file, bag_file, keep_saccades):
    timeline = []
    fixations = {'red': [], 'yellow':[], 'green':[], 'other':[]}
    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                #vid_time = d['ts'] - d['vts']
                vid2ts[d['vts']] = d['ts']

        # TODO: if multiple detections for same time stamp?
        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')



    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        model.append((m,c))
        a = all_vts[i]


    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()
    print('reading video file')

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    videoframe2trackerts = []
    gaze_pts = []

    while success:          
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts, _ = takeClosest(all_ts,ts)
        videoframe2trackerts.append(tracker_ts)

        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
        gaze_pts.append(gaze_coords)
        #h,s,v = img[gaze_coords[1]][gaze_coords[0]]
        b, g, r = img[gaze_coords[1]][gaze_coords[0]]
        instant_color = [r/255.0,g/255.0,b/255.0]
        timeline.append(instant_color)

        last_gaze_pt = gaze_coords

        count += 1
        success, img = vidcap.read()

    vidcap.release()
    cv2.destroyAllWindows()

    saccade_indices = []
    if not keep_saccades:
        timeline, saccade_indices = remove_saccades(gaze_pts, timeline, fps)

    # find segmentation points on bagfile
    keyframes = []
    open_keyframe = []
    record_k, record_o = False, False
    bag = rosbag.Bag(bag_file)
    print(bag_file)
    if bag.get_message_count('/gaze_tracker')!=0:       # gaze_tracker topic was recorded
        for topic, msg, t in bag.read_messages(topics=['/gaze_tracker','/log_KTframe']):
            #if('vts' in msg.data):
            #print topic
            if (topic=='/log_KTframe'):
                if("Recorded keyframe" in msg.data):
                    record_k = True
                if("Open" in msg.data):
                    record_o = True
            if (topic == '/gaze_tracker'):
                if(record_k == True):                   
                    if('gp' in msg.data):                   
                        gaze_msg = msg.data
                        s = gaze_msg.find('"ts":')
                        e = gaze_msg.find(',')
                        gaze_ts = gaze_msg[s+5:e]
                        # print('gaze_ts:',gaze_ts)
                        tracker_ts, frame_idx = takeClosest(videoframe2trackerts,int(gaze_ts))
                        # print('tracker_ts:',tracker_ts)
                        keyframes.append(frame_idx)
                        record_k = False
                if(record_o == True):
                    if('gp' in msg.data):       
                        gaze_msg = msg.data
                        s = gaze_msg.find('"ts":')
                        e = gaze_msg.find(',')
                        gaze_ts = gaze_msg[s+5:e]
                        # print('gaze_ts:',gaze_ts)
                        tracker_ts, frame_idx = takeClosest(videoframe2trackerts,int(gaze_ts))
                        # print('tracker_ts:',tracker_ts)
                        open_keyframe.append(frame_idx)
                        record_o = False

    # print(keyframes)
    # print(open_keyframe)
    # print(len(videoframe2trackerts),videoframe2trackerts[-1])
    bag.close()

    return timeline, keyframes, open_keyframe, saccade_indices



def remove_saccades(gaze_pts, color_timeline, fps):
    speed = []
    saccade_indices = []
    speed.append(0)
    dt = 1.0/fps
    for i in range(1,len(gaze_pts)):
        g = gaze_pts[i]
        prev_g = gaze_pts[i-1]
        s = (math.sqrt(math.pow(g[0]-prev_g[0],2)+math.pow(g[1]-prev_g[1],2)))/dt
        # print(s)
        if s>200:
            # print('*****',s)
            color_timeline[i] = [1.0, 1.0, 1.0]
            saccade_indices.append(i)
        # print(s)
    return color_timeline, saccade_indices


def get_cumulative_gaze_dist(data, video_file):
    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                #vid_time = d['ts'] - d['vts']
                vid2ts[d['vts']] = d['ts']

        # TODO: if multiple detections for same time stamp?
        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')

    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        model.append((m,c))
        a = all_vts[i]


    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()
    print('reading video file')

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    gaze_pts = []

    current_dist = 0
    cumulative_dist = [0]
    tracker_ts, _ = takeClosest(all_ts,all_vts[0])
    gx_p, gy_p = gp[tracker_ts]

    while success:  
        # print('reading frame %d', count)      
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts, _ = takeClosest(all_ts,ts)

        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
        gaze_pts.append(gaze_coords)

        gx, gy = gaze_coords
        d = math.sqrt(math.pow(gx-gx_p,2)+math.pow(gy-gy_p,2))
        current_dist = current_dist + d
        cumulative_dist.append(current_dist)
        gx_p, gy_p = gx, gy
        
        count += 1
        success, img = vidcap.read()

    vidcap.release()
    cv2.destroyAllWindows()

    return cumulative_dist

def get_color_name(hsv):

    color_ranges = {
        'red':   [[160,90,30],[190,190,135]],
        'green': [[83,55,25],[115,110,125]],
        'yellow': [[5,100,40],[35,165,170]],
        'blue': [[95,75,55],[120,120,160]],
        'orange': [[0,145,65],[190,230,175]],
        'purple': [[115,50,10],[145,150,125]],
        'black': [[0,0,0],[180,255,40]],
        'white': [[0,0,170],[180,255,255]]
    }

    color_val = {
        'black': (0,0,0),
        'white': (255,255,255),
        'red': (0,0,255),
        'green': (0,255,0),
        'yellow': (0,255,255),
        'blue': (255,0,0),
        'orange': (165,0,255),
        'purple': (32,240,160)
    }
        

    h,s,v = hsv
    color = ''
    value = None
    for i, (n,r) in enumerate(color_ranges.items()):
        # print(n, r[0][0], r[1][0])
        if h>=r[0][0] and h<=r[1][0]:
            if s>=r[0][1] and s<=r[1][1]:
                if v>=r[0][2] and v<=r[1][2]:
                    color = n 
                    value = color_val[n]

    pasta_color_range = [[0,30,0],[40,130,150]]
    p = pasta_color_range
    if color=='':
        if h>=p[0][0] and h<=p[1][0]:
            if s>=p[0][1] and s<=p[1][1]:
                if v>=p[0][2] and v<=p[1][2]:
                    color = 'pasta'
                    value = color_val['yellow']

    return color, value


def get_color_name_from_hist(gaze_coords, img_hsv, radius):
    color_hist ={
        'orange': 0,
        'yellow': 0,
        'red': 0,
        'green': 0,
        'black': 0,
        'pasta': 0,
        'blue': 0,
        'purple': 0,
        'other': 0
    }

    color_val = {
        'black': (0,0,0),
        'red': (0,0,255),
        'green': (0,255,0),
        'yellow': (0,255,255),
        'pasta': (0,255,255),
        'blue': (255,0,0),
        'orange': (165,0,255),
        'purple': (32,240,160),
        'other': (192,192,192)
    }

    x, y = gaze_coords
    hsv = img_hsv[y][x]
    h,s,v = hsv
    color = ''
    value = None

    # pixels in the image which lie inside a circle of given radius
    min_x, max_x = max(0,x-radius), min(1920, x+radius)
    min_y, max_y = max(0,y-radius), min(1080, y+radius)
    for i,j in zip(range(min_x,max_x), range(min_y,max_y)):
        d = math.pow((i-x),2)+ math.pow((j-y),2)
        if d<= math.pow(radius,2):
            curr_hsv= img_hsv[j][i]
            current_color, _ = get_color_name(curr_hsv)
            if current_color in color_hist.keys():
                color_hist[current_color] += 1
            else:
                color_hist['other'] += 1

    max_val = 0
    max_color = ''
    for key,val in color_hist.items():
        # print(val)
        if val>max_val:
            max_val = val
            max_color = key


    # do not assign other color if relevant colors are present
    second_max_val = 0
    second_max_color = ''
    if max_color=='other':
        for key,val in color_hist.items():
            if key=='other':
                continue
            else:
                if val>second_max_val:
                    second_max_val = val
                    second_max_color = key
        if second_max_val>5:
            max_color = second_max_color
            max_val = second_max_val

    value = color_val[max_color]
    return max_color, value



def find_saccades(gaze_pts, fps):
    speed = []
    saccade_indices = []
    speed.append(0)
    dt = 1.0/fps
    for i in range(1,len(gaze_pts)):
        g = gaze_pts[i]
        prev_g = gaze_pts[i-1]
        s = (math.sqrt(math.pow(g[0]-prev_g[0],2)+math.pow(g[1]-prev_g[1],2)))/dt
        # print(s)
        if s>800:
            # print('*****',s)
            saccade_indices.append(i)
        # print(s)
    return saccade_indices



# returns a list of rgb color values for gaze point for each video frame
def get_hsv_color_timeline(data, video_file):
    timeline = []
    # fixations = {'red': [], 'yellow':[], 'green':[], 'other':[]}
    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                #vid_time = d['ts'] - d['vts']
                vid2ts[d['vts']] = d['ts']

        # TODO: if multiple detections for same time stamp?
        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')


    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        model.append((m,c))
        a = all_vts[i]

    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()
    # img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    print('reading video file')

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    gaze_pts = []

    while success:  
        # print(count)  
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)   
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts,  _ = takeClosest(all_ts,ts)
        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
        gaze_pts.append(gaze_coords)

        h,s,v = img_hsv[gaze_coords[1]][gaze_coords[0]]
        # b, g, r = img[gaze_coords[1]][gaze_coords[0]]
        # instant_color = [r/255.0,g/255.0,b/255.0]
        instant_color = [h, s, v]
        timeline.append(instant_color)

        # last_gaze_pt = gaze_coords

        count += 1
        success, img = vidcap.read()

        

    vidcap.release()
    cv2.destroyAllWindows()

    saccade_indices = []
    # if not keep_saccades:
    saccade_indices = find_saccades(gaze_pts, fps)
    # fixations = find_fixations(gaze_pts, fps, saccade_indices)

    return timeline, saccade_indices


# returns a list of frame indices corresponding to the annotated KF for video demonstrations
def get_video_keyframes(user_id, seg, video_file, video_kf_file):
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('read video file')

    vidcap.release()
    cv2.destroyAllWindows()

    # read video files
    with open(video_kf_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    # print(content)
    print('read text file')

    # find segmentation points in video file
    keyframes = {
        'Start': [],
        'Open': [],
        'Stop': []
    }

    kf_type = {
        1: 'Start',
        2: 'Open',
        3: 'Stop',
        4: 'Start',
        5: 'Open',
        6: 'Stop'
    }

    if seg%2==0:
        r = [4, 5, 6]
    else:
        r = [1, 2, 3] 
    for kf in content:                  
        data = kf.split(' ')
        # print(data)
        user = data[0]
        if(user == user_id):
            for i in r:
                d = data[i]
                # print(d)
                if(d=='end'):
                    frame_idx = length
                else:
                    kf_time = float(d)
                    frame_idx = math.floor(kf_time*fps)
                k = kf_type[i]
                keyframes[k].append(int(frame_idx))

    print('Found start and stop keyframe indices')
    return keyframes

def read_json(data_dir):
    data = []
    files = os.listdir(data_dir)
    
    for file in files:
        if (file.endswith("json.gz")):
            with gzip.open(data_dir+'/'+file, "rb") as f:
                data=f.readlines()
            
                for r in range(len(data)):
                    row = data[r]
                    data[r] = ast.literal_eval(row.strip('\n'))

    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                vid2ts[d['vts']] = d['ts']

        # TODO: if multiple detections for same time stamp?
        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')

    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        # print("Line Solution is ts = {m}vts + {c}".format(m=m,c=c))
        model.append((m,c))

    return data, gp, model, all_vts


def filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx):
    print('filtering fixations')
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # print fps     #25 fps
    success, img = vidcap.read()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter('../../data/gaze_pouring_gaze_filtered_user6_exp3.avi',fourcc,fps,(1920,1080))

    # video_fixation_count = {
    #   'red': 0,
    #   'yellow': 0,
    #   'blue': 0,
    #   'green': 0,
    #   'other': 0
    # }

    KT_fixation_count = {
        'red': 0,
        'orange': 0,
        'blue': 0,
        'green': 0,
        'black': 0,
        'yellow': 0,
        'purple': 0,
        'other': 0,
        'pasta': 0
    }

    fixation_count = KT_fixation_count

    # if demo_type=='k':
    #   fixation_count = KT_fixation_count
    # else if demo_type=='v':
    #   fixation_count = video_fixation_count



    all_ts = sorted(gp.keys())
    total_count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    window = []
    win_size = 3
    radius = 100
    valid_count = 0
    # print(start_idx, end_idx)
    while success:  
        # print(count)
        if total_count<start_idx or total_count>end_idx:
            total_count += 1
            success, img = vidcap.read()
            continue

        frame_ts = int((total_count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c

        tracker_ts,_ = takeClosest(all_ts,ts)

        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))

        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # hsv = img_hsv[gaze_coords[1]][gaze_coords[0]]

        color_name, color_value = get_color_name_from_hist(gaze_coords, img_hsv, radius)
        window.append(color_name)
        if(len(window)>win_size):
            del window[0]

        font = cv2.FONT_HERSHEY_SIMPLEX
        if total_count not in saccade_indices:
            # might be a fixation
            fixation = True
            for det_c in window:
                if det_c!=color_name:
                    fixation=False
            if(fixation):
                fixation_count[color_name] += 1

        valid_count += 1
        total_count += 1
        success, img = vidcap.read()

    cv2.destroyAllWindows()
    print total_count
    for f in fixation_count:
        fixation_count[f] = fixation_count[f]*100.0/valid_count

    return fixation_count



def get_kt_keyframes(all_vts, model, gp, video_file, bag_file):
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()
    print('reading video file')

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    videoframe2trackerts = []
    gaze_pts = []

    while success:          
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts, _ = takeClosest(all_ts,ts)
        videoframe2trackerts.append(tracker_ts)

        count += 1
        success, img = vidcap.read()

    vidcap.release()
    cv2.destroyAllWindows()

    # find segmentation points on bagfile
    all_keyframe_indices = []
    record_k = False
    bag = rosbag.Bag(bag_file)
    print(bag_file)
    if bag.get_message_count('/gaze_tracker')!=0:       # gaze_tracker topic was recorded
        for topic, msg, t in bag.read_messages(topics=['/gaze_tracker','/log_KTframe']):
            #if('vts' in msg.data):
            #print topic
            if (topic=='/log_KTframe'):
                # print(msg.data)
                if("Recorded keyframe" in msg.data):
                    record_k = True

                    kf_type = 'Other'

                if("Open" in msg.data):
                    record_k = True
                    kf_type = 'Open'
                # if("Close" in msg.data):
                #     record_k = True
                #     kf_type = 'Close'

            if (topic == '/gaze_tracker'):
                if(record_k == True):                   
                    if('gp' in msg.data):                   
                        gaze_msg = msg.data
                        s = gaze_msg.find('"ts":')
                        e = gaze_msg.find(',')
                        gaze_ts = gaze_msg[s+5:e]
                        tracker_ts, frame_idx = takeClosest(videoframe2trackerts,int(gaze_ts))
                        all_keyframe_indices.append(frame_idx)
                        record_k = False

    # print(keyframes)
    # print(open_keyframe)
    # print(len(videoframe2trackerts),videoframe2trackerts[-1])
    bag.close()
    return all_keyframe_indices