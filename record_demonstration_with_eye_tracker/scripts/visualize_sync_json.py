import cv2
import ast 

vidcap = cv2.VideoCapture('../data/fullstream.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS)
print fps 	#25 fps
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

with open ("../data/livedata.json", "r") as myfile:
	data=myfile.readlines()

for r in range(len(data)):
	row = data[r]
	data[r] = ast.literal_eval(row.strip('\n'))

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

RE_pd_list = sorted(right_eye_pd.iterkeys())
LE_pd_list = sorted(left_eye_pd.iterkeys())

gp_list = sorted(gp.iterkeys())
print(len(gp_list))    # Total Number of gaze points detected during the video (sampling rate 50Hz)

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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('./videos/001_output.mp4',fourcc, 29.0, size, False)
video = cv2.VideoWriter('video.avi',fourcc,fps,(1920,1080))

vidcap = cv2.VideoCapture('../data/fullstream.mp4')
success,img = vidcap.read()
i = 0
while success:
	
	#for i,img in enumerate(imgs):
	print(i)
	ts = frame2ts[i]
	all_tracker_ts = sorted(vid2ts.keys())
	#print(all_tracker_ts)
	print(ts)
	ts_pos = takeClosest(all_tracker_ts, ts)
	print(ts_pos)
	#print(vid2ts)
	#if(vid2ts[0] in gp_list):
	#	gaze = gp[vid2ts[all_tracker_ts[ts_pos]]]
	gaze = gp[gp_list[i]]
	print(gaze)
	cv2.circle(img,(int(gaze[0]*1920), int(gaze[1]*1080)), 5, (0,255,0), -1)
	video.write(img)
	i = i + 1
	success,img = vidcap.read()
	print(success)
cv2.destroyAllWindows()
video.release()