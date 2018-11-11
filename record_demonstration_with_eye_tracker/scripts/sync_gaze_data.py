# DO NOT USE - SLOWS DOWN ALIENWARE

import cv2
import ast 

vidcap = cv2.VideoCapture('../data_old/fullstream.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS)
print fps 
success,image = vidcap.read()

count = 0
imgs = [] 		# list of image frames
frame2ts = []	# corresponding list of video time stamp values in microseconds
success = True
while success:
  #cv2.imwrite("data/imgs/%d.jpg" % count, image)     # save frame as JPEG file   
  imgs.append(image)       # ************HOGS ALL THE MEMORY******************
  success,image = vidcap.read()
  frame2ts.append(int((count/fps)*1000000))
  #print('Read a new frame: ', success)
  count += 1

with open ("../data_old/livedata.json", "r") as myfile:
    data=myfile.readlines()

for r in range(len(data)):
	row = data[r]
	data[r] = ast.literal_eval(row.strip('\n'))

vid2ts = {}		# dictionary mapping video time to time stamps in json
right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

for d in data:
	if 'vts' in d and d['s']==0:
		if d['vts'] == 0:
			#start_frame_ts = d['ts']
			print str(d['vts']) + '\t\t' + str(d['ts']/1000000.)
			vid2ts[d['vts']] = d['ts']
		else:
			#vid_time = d['ts'] - d['vts']
			vid2ts[d['vts']] = d['ts']
			print str(d['vts']/1000000.)  + '\t\t' + str(d['ts']/1000000.)

	# TODO: if multiple detections for same time stamp?
	if 'pd' in d and d['s']==0 and d['eye']=='right':
		right_eye_pd[d['ts']] = d['pd']
	if 'pd' in d and d['s']==0 and d['eye']=='left':
		left_eye_pd[d['ts']] = d['pd']

	if 'gp' in d and d['s']==0 :
		gp[d['ts']] = d['gp']	#list of 2 coordinates


RE_pd_list = sorted(right_eye_pd.iterkeys())
LE_pd_list = sorted(left_eye_pd.iterkeys())

gp_list = sorted(gp.iterkeys())
#print(len(gp_list)) 	# Total Number of gaze points detected during the video (sampling rate 50Hz)

# how many img frames between two keypoints?
# how many gaze data points between two keypoints?
# linear interpolation
keypoint_imgs2ts = {}
keypoint_imgs2ts[0] = vid2ts[0]
for vid_time in sorted(vid2ts.iterkeys()):
	#print vid_time
	if vid_time!=0:
		gaze_idx = next((index for (index, d) in enumerate(data) if d['ts'] == vid2ts[vid_time]), None)
		gaze_ts = vid2ts[vid_time]
		try:
			img_idx = filter(lambda i: (frame2ts[i]<=vid_time and vid_time<frame2ts[i+1], frame2ts) )	#needs to change
			print img_idx
			#print str(img_idx) + '\t\t' + str(gaze_idx)
			keypoint_imgs2ts[img_idx] = gaze_ts
			#print str(img_idx) + '\t' + str(gaze_ts/1000000.)
		except:
			print 'failed!'
#TODO: last vts message needs to be synced with an image frame
#keypoint_imgs2ts[len(frame2ts)-1] = vid2ts[]

# For every remaining img frame (apart from keypoints), interpolate linearly for the two keypoints around it
imgframe2ts = {}
curr_keypoint = 0
for img_idx in range(len(frame2ts)-1):
	#if img frame is a keypoint frame, then directly use the matched value from keypoint_img2ts
	if img_idx in keypoint_imgs2ts:
		imgframe2ts[img_idx] = keypoint_imgs2ts[img_idx]
		curr_keypoint = img_idx
		curr_idx = sorted(keypoint_imgs2ts.keys()).index(img_idx)
	#else use line fitting, extract ts which is closest to predcited linear interpolation
	else:
		next_idx = sorted(keypoint_imgs2ts.keys())[curr_idx+1]
		slope = float(keypoint_imgs2ts[next_idx] - keypoint_imgs2ts[curr_idx])/(next_idx - curr_idx)
		interpolated_ts = keypoint_imgs2ts[curr_idx] + slope*(img_idx-curr_idx)
		matching_ts = min(range(len(gp_list)-1), key=lambda i: gp_list[i]<=interpolated_ts and interpolated_ts<gp_list[i+1])	# also needs to change
		imgframe2ts[img_idx] = matching_ts
#print '**********'
#print str(len(imgs)) + '\t\t' + str(len(data))