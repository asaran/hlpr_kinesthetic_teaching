import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

#bag_usb = rosbag.Bag('2018-05-09-17-33-46.bag')
bag_gaze = rosbag.Bag('/home/akanksha/Documents/rss 2019 data/KT7/bags/kt7-s-p1.bag')
#for topic, msg, t in bag_usb.read_messages(topics=['chatter', 'numbers']):
#    print msg

bridge = CvBridge()

count = 0
save_dir =  '/home/akanksha/Documents/rss 2019 data/KT7/imgs/p1'
for topic, msg, t in bag_gaze.read_messages(topics=['/kinect/qhd/image_color_rect/compressed']):
	if topic == "/kinect/qhd/image_color_rect/compressed":
		try:
			cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError, e:
			print e
		count = count + 1
		timestr = "%.6f" % msg.header.stamp.to_sec()
		image_name = str(save_dir)+"/"+str(count)+".png"
		#image_name = str(save_dir)+"/"+timestr+"_left"+".pgm"
		cv2.imwrite(image_name, cv_image)
	
bag_gaze.close()