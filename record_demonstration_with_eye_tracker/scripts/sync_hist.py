import cv2
import ast
from bisect import bisect_left
from numpy import ones,vstack
from numpy.linalg import lstsq
import numpy as np
import math

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
		return myList[0]
	if pos == len(myList):
		return myList[-1]
	before = myList[pos - 1]
	after = myList[pos]
	if after - myNumber < myNumber - before:
	   return after
	else:
	   return before

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
		m, c = lstsq(A, y_coords, rcond=None)[0]
		#print("Line Solution is ts = {m}vts + {c}".format(m=m,c=c))
		model.append((m,c))
		a = all_vts[i]


	vidcap = cv2.VideoCapture(video_file)
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	#print fps 	#25 fps
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
		tracker_ts = takeClosest(all_ts,ts)
		#print('tracker_ts_pos: ', tracker_ts)
		gaze = gp[tracker_ts]
		gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
		#print(gaze_coords)
		h,s,v = img[gaze_coords[1]][gaze_coords[0]]
		if(count==0):
		# 	last_fixation_color = (h,s,v)
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
		# 	t = t + int((1.0/fps)*1000)
			
		# else:
		# 	if(t>100):
		# 		if is_known_color(mean_hsv) is not None:
		# 			fixations[is_known_color(mean_hsv)].append(t)
		# 		else:
		# 			fixations['other'].append(t)
		# 	t = 0

		# #cv2.circle(img,(int(gaze[0]*1920), int(gaze[1]*1080)), 5, (0,255,0), -1)
		# #video.write(img)

		# last_fixation_color = (h,s,v)
		last_gaze_pt = gaze_coords

		count += 1
		success, img = vidcap.read()



	cv2.destroyAllWindows()
	#video.release()
	vidcap.release()
	return hist