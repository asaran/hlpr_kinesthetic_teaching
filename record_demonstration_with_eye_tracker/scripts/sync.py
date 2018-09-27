import cv2
import ast
from bisect import bisect_left


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


def sync_func(data, video_file):
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
		print("Line Solution is ts = {m}vts + {c}".format(m=m,c=c))
		model.append((m,c))



	vidcap = cv2.VideoCapture(video_file)
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	print fps 	#25 fps
	success, img = vidcap.read()

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoWriter('gaze.avi',fourcc,fps,(1920,1080))

	all_ts = sorted(gp.keys())
	count = 0
	imgs = []       # list of image frames
	frame2ts = []   # corresponding list of video time stamp values in microseconds
	while success:	
		print(count)
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
		cv2.circle(img,(int(gaze[0]*1920), int(gaze[1]*1080)), 5, (0,255,0), -1)
		video.write(img)

		count += 1
		success, img = vidcap.read()

	cv2.destroyAllWindows()
	video.release()