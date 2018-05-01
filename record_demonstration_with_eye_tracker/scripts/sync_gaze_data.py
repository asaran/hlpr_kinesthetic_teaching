import cv2
import ast 

vidcap = cv2.VideoCapture('data/fullstream.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS)
print fps 
success,image = vidcap.read()

count = 0
imgs = [] 		# list of image frames
frame2ts = []	# corresponding list of video time stamp values in microseconds
success = True
while success:
  #cv2.imwrite("data/imgs/%d.jpg" % count, image)     # save frame as JPEG file   
  imgs.append(image)   
  success,image = vidcap.read()
  frame2ts.append(int((count/fps)*1000000))
  #print('Read a new frame: ', success)
  count += 1

#print frame2ts
#data = json.load(open('data/livedata.json'))
#pprint(data)

with open ("data/livedata.json", "r") as myfile:
    data=myfile.readlines()

for r in range(len(data)):
	row = data[r]
	data[r] = ast.literal_eval(row.strip('\n'))

vid2ts = {}
#print data
for d in data:
	if 'vts' in d and d['s']==0:
		if d['vts'] == 0:
			start_frame_ts = d['ts']
			print str(d['vts']) + '\t\t' + str(d['ts']/1000000.)
			vid2ts[d['vts']] = d['ts']
		else:
			vid_time = d['ts'] - d['vts']
			vid2ts[d['vts']] = d['ts']
			print str(d['vts']/1000000.)  + '\t\t' + str(d['ts']/1000000.)


# how many img frames between two keypoints?
# how many gaze data points between two keypoints?
# linear interpolation
prev_img_idx = 0
prev_gaze_idx = next((index for (index, d) in enumerate(data) if d['ts'] == vid2ts[0]), None)
prev_gaze_ts = vid2ts[0]
for vid_time in sorted(vid2ts.iterkeys()):
	if vid_time!=0:
		gaze_idx = next((index for (index, d) in enumerate(data) if d['ts'] == vid2ts[vid_time]), None)
		gaze_ts = vid2ts[vid_time]
		img_idx = min(range(len(frame2ts)-1), key=lambda i: frame2ts[i]<=vid_time and vid_time>=frame2ts[i+1])
		print str(img_idx) + '\t\t' + str(gaze_idx)

print '**********'
print str(len(imgs)) + '\t\t' + str(len(data))