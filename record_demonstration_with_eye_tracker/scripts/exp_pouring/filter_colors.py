import cv2
import numpy as np

while(1):
	frame = cv2.imread('../../data/imgs_pouring/300.png')
	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# define range of blue color in HSV
	lower_blue = np.array([110,50,50])
	upper_blue = np.array([130,255,255])

	lower_red = np.array([170,65,60])
	upper_red = np.array([190,180,90])

	lower_yellow = np.array([10,130,110])
	upper_yellow = np.array([25,160,150])

	# hsv
	lower_black = np.array([0,0,0])
	upper_black = np.array([180,255,40])

	# rgb
	lower_black_rgb = np.array([0,0,0])
	upper_black_rgb = np.array([50,50,50])

	# hsv white
	lower_white = np.array([0,0,140])
	upper_white = np.array([180,255,255])

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower_white, upper_white)
	# mask = cv2.inRange(frame, lower_black_rgb, upper_black_rgb)
	grey_frame = np.zeros(frame.shape, np.uint8)
	grey_frame[:,:] = (169,169,169)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame,frame, mask=mask)

	# resize frames for display
	frame = cv2.resize(frame, (640, 480)) 
	mask = cv2.resize(mask, (640, 480)) 
	res = cv2.resize(res, (640, 480)) 

	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	cv2.imshow('res',res)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
