import cv2
import numpy as np

while(1):
	frame = cv2.imread('imgs/642.png')
	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# define range of blue color in HSV
	lower_blue = np.array([110,50,50])
	upper_blue = np.array([130,255,255])

	lower_red = np.array([170,65,60])
	upper_red = np.array([190,180,90])

	lower_yellow = np.array([10,130,110])
	upper_yellow = np.array([25,160,150])

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower_red, upper_red)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame,frame, mask= mask)

	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	cv2.imshow('res',res)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()

print('red plate: ',hsv[620][1366])

print('red plate: ',hsv[619][1293])
print('red plate: ',hsv[641][1370])
print('red plate: ',hsv[549][1380])
print('red plate: ',hsv[710][1323])
print('red plate: ',hsv[717][1286])
print('red plate: ',hsv[478][1151])

print('orange cup: ', hsv[775][1616])
print('orange cup: ', hsv[772][1671])
print('orange cup: ', hsv[809][1648])

print('yellow plate: ', hsv[536][649]) 
print('yellow plate: ', hsv[458][593]) 
print('yellow plate: ', hsv[525][592]) 
print('yellow plate: ', hsv[507][661]) 

print('green spoon: ', hsv[230][950])
print('green spoon: ', hsv[236][1049])
print('green spoon: ', hsv[253][989])
print('green spoon: ', hsv[384][971])


print('blue handle: ', hsv[513][936])
# 936 513 blue handle

#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(frame,(x,y),100,(255,0,0),-1)
        ix,iy = x,y

# Create a black image, a window and bind the function to window
#img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',frame)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print ix,iy
cv2.destroyAllWindows()