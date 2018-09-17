import rosbag
#bag_usb = rosbag.Bag('2018-05-09-17-33-46.bag')
bag_gaze = rosbag.Bag('../data/2018-05-09-17-33-53.bag')
#for topic, msg, t in bag_usb.read_messages(topics=['chatter', 'numbers']):
#    print msg

for topic, msg, t in bag_gaze.read_messages(topics=['/gaze_tracker']):
	if('vts\"' in msg.data):
		print t, msg
bag_gaze.close()