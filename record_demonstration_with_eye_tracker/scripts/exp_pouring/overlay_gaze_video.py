from numpy import ones,vstack
from numpy.linalg import lstsq
import cv2
import ast 
from utils import get_color_name

# my_dir = '../../data/pouring/experts/KT6/5fyyvco/segments/1/'
my_dir = '../../data/pouring/experts/KT12/bmzotwj/segments/1/'

with open(my_dir+"livedata.json", "r") as myfile:
	data=myfile.readlines()

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
	print("Line Solution is ts = {m}vts + {c}".format(m=m,c=c))
	model.append((m,c))


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


vidcap = cv2.VideoCapture(my_dir+'fullstream.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS)
print fps 	#25 fps
success, img = vidcap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video = cv2.VideoWriter('../../data/gaze_pouring_color_filtered_user6_exp4_again.avi',fourcc,fps,(1920,1080))
video = cv2.VideoWriter('../../data/gaze_pouring_color_filtered_user12_exp1.avi',fourcc,fps,(1920,1080))

all_ts = sorted(gp.keys())
count = 0
imgs = []       # list of image frames
frame2ts = []   # corresponding list of video time stamp values in microseconds
while success:	
	# print(count)
	frame_ts = int((count/fps)*1000000)
	frame2ts.append(frame_ts)

	less = [a for a in all_vts if a<=frame_ts]
	idx = len(less)-1

	if idx<len(model):
		m,c = model[idx]
	else:
		m,c = model[len(model)-1]
	ts = m*frame_ts + c

	tracker_ts = takeClosest(all_ts,ts)

	gaze = gp[tracker_ts]
	gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))

	img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	hsv = img_hsv[gaze_coords[1]][gaze_coords[0]]
	font = cv2.FONT_HERSHEY_SIMPLEX
	color_name, color_value = get_color_name(hsv)
	
	# if(color_name!=''):
	# 	print(color_name)
		# cv2.putText(img, color_name, (1430, 250), font, 1.8, color_value, 5, cv2.LINE_AA)

	# print(hsv)
	# cv2.putText(img, str(hsv), (230, 250), font, 1.8, (255, 255, 0), 5, cv2.LINE_AA)

	cv2.circle(img,gaze_coords, 50, (255,255,255), thickness=15)
	img = cv2.resize(img,None,fx=0.5,fy=0.5)	#resize image
	video.write(img)
	# cv2.imwrite('../../data/imgs_pouring_KT/'+str(count)+'.png', img)

	cv2.imwrite('../../data/video_imgs/'+str(count)+'.png', img)
	count += 1
	success, img = vidcap.read()

cv2.destroyAllWindows()
video.release()