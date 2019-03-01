import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default
chosen_colors = []

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,y, image_hsv.shape[0], image_hsv.shape[1], x/float(image_hsv.shape[1]), y/float(image_hsv.shape[0]))
        pixel = image_hsv[y,x]

        chosen_colors.append(pixel)

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel, lower, upper)

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)

def main():
    import sys
    global image_hsv, pixel # so we can use it in mouse callback

    image_src = cv2.imread(sys.argv[1])  # pick.py my.png
    if image_src is None:
        print ("the image read is None............")
        return
    # cv2.imshow("bgr",image_src)

    ## NEW ##
    # cv2.namedWindow('hsv')
    cv2.namedWindow('bgr')
    while(1):
        cv2.setMouseCallback('bgr', pick_color)

        # now click into the hsv img , and look at values:
        image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv",image_hsv)
        cv2.imshow("bgr",image_src)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

    for col in chosen_colors:
        h,s,v = col


if __name__=='__main__':
    main()