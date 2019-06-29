#!/usr/bin/env python

from hlpr_perception_msgs.msg import LabeledObjects, SegClusters
from std_msgs.msg import String, Float32MultiArray
from hlpr_color_object_labeler.object_filters import *
import rospy
import image_geometry 
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
import tf
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

class ObjectsLocator:
    def __init__(self):
        rospy.init_node('object_locator')
         
        self.labels_topic = rospy.get_param('~labels_topic', '/beliefs/labels')
        self.obj_filters = object_filters_from_yaml(rospy.get_param('/hlpr_color_object_labeler/task_yaml'))
        self.camera_data = None
        rospy.Subscriber('/kinect/qhd/camera_info', CameraInfo, self.set_cam_data)
        self.image_sub = rospy.Subscriber("/kinect/qhd/image_color",Image,self.image_callback)

        self.listener = tf.TransformListener()
        self.listener2 = tf.TransformListener()
        self.listener3 = tf.TransformListener()
        self.bridge = CvBridge()
        self.blue_x, self.blue_y = None, None
        self.red_x, self.red_y = None, None
        self.plane_info = None
        self.place_pub = None
        self.plate = None

    def image_callback(self,data):
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        (rows,cols,channels) = cv_image.shape
        if self.blue_x!=None and self.blue_y!=None :
          cv2.circle(cv_image, (int(self.blue_x),int(self.blue_y)), 10, (255,0,0), -1)

        if self.red_x!=None and self.red_y!=None :
          cv2.circle(cv_image, (int(self.red_x),int(self.red_y)), 10, (0,0,255), -1)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)


    def listen_labels(self, msg):
        # print(self.obj_filters)
        for name, obj_filter in self.obj_filters.iteritems():
            obj_idx = obj_filter(msg)
            if obj_idx:
                obj = msg.objects[obj_idx[0]]
                print(name) # red_bowl, blue_bowl
                # print(obj.obb.bb_center) # in kinect_ir_optical_frame

                obj_center = obj.obb.bb_center
                obj_center_point = Point(obj_center.x, obj_center.y, obj_center.z)
                print(obj_center_point)

                now = rospy.Time.now()
                object_ir_frame = PointStamped()
                ir_frame = self.camera_data.header.frame_id
                # print(type(ir_frame))
                object_ir_frame.header.frame_id = ir_frame
                object_ir_frame.header.stamp = now
                object_ir_frame.point.x = obj_center_point.x
                object_ir_frame.point.y = obj_center_point.y
                object_ir_frame.point.z = obj_center_point.z

                # transform point from kinect_ir_optical_frame to kinect_rgb_optical_frame
                self.listener.waitForTransform('kinect_rgb_optical_frame',ir_frame, now, rospy.Duration(10.0))
                rgb_frame = self.listener.transformPoint('kinect_rgb_optical_frame',object_ir_frame)
                x_0 = np.array([rgb_frame.point.x,rgb_frame.point.y,rgb_frame.point.z])
                print(x_0)
                
                cam_model = image_geometry.PinholeCameraModel()
                # print(self.camera_data)
                cam_model.fromCameraInfo(self.camera_data)

                # project point from 3D world to rgb image
                # u,v = cam_model.project3dToPixel([obj_center.x,obj_center.y,obj_center.z])
                u,v = cam_model.project3dToPixel([x_0[0],x_0[1],x_0[2]])
                print(u,v)

                # TODO: verify u,v overlay on the image correctly
                if name=="blue_bowl":
                    self.blue_x, self.blue_y = u,v
                if name=="red_bowl":
                    self.red_x, self.red_y = u,v
                    self.plate = obj_center_point

        # Run BIRL and get placement location on image
        place_x = self.red_x 
        place_y = self.red_y 
        # placePoint = PointStamped()

        # Transform placement location to 3D vector in rgb frame
        vec = cam_model.projectPixelTo3dRay((place_x,place_y))

        
        # Convert placement vector to IR frame where the table plane comes from
        vec_rgb_frame = PointStamped()
        vec_rgb_frame.header.frame_id = 'kinect_rgb_optical_frame'
        vec_rgb_frame.header.stamp = now
        vec_rgb_frame.point.x = vec[0]
        vec_rgb_frame.point.y = vec[1]
        vec_rgb_frame.point.z = vec[2]
        self.listener2.waitForTransform('kinect_ir_optical_frame','kinect_rgb_optical_frame', now, rospy.Duration(10.0))
        ir_vec = self.listener2.transformPoint('kinect_ir_optical_frame',vec_rgb_frame)
        x_1 = np.array([ir_vec.point.x,ir_vec.point.y,ir_vec.point.z])

        # Find intersection of 3D vector in direction of placement point and table plane to get the actual 3D location for placement (in IR frame)


        # Convert final 3D placement point to right_link_base frame
        place_point = PointStamped()
        place_point.header.frame_id = 'kinect_ir_optical_frame'
        place_point.header.stamp = now
        place_point.point.x = self.plate.x
        place_point.point.y = self.plate.y
        place_point.point.z = self.plate.z
              
        self.listener3.waitForTransform('right_link_base','kinect_ir_optical_frame', now, rospy.Duration(10.0))
        base_frame = self.listener3.transformPoint('right_link_base',place_point)
        

        # publish place point
        place_loc = Float32MultiArray()
        place_loc.data = [base_frame.point.x, base_frame.point.y, base_frame.point.z]
        self.place_pub.publish(place_loc)

        br = tf.TransformBroadcaster()
        br.sendTransform((base_frame.point.x, base_frame.point.y, base_frame.point.z),
                         (0.305, 0.588, 0.389, 0.640),
                         rospy.Time.now(),
                         "spoon",
                         "right_link_base")


    # def handle_turtle_pose(msg, turtlename):
        


    def set_cam_data(self,msg):
        self.camera_data = msg

    def set_plane_info(self,msg):
        self.plane_info = msg.plane.data
        print('*****plane info:*****')
        print(msg.plane.data)
        print('plane info end')

    def run(self):
        
        # Subscribe to Kinect v2 sd camera_info to get image frame height and width
        rospy.Subscriber('/kinect/qhd/camera_info', CameraInfo, self.set_cam_data)
        rospy.Subscriber(self.labels_topic, LabeledObjects, self.listen_labels)
        self.image_sub = rospy.Subscriber("/kinect/qhd/image_color",Image,self.image_callback)
        rospy.Subscriber('/beliefs/clusters', SegClusters, self.set_plane_info)
        # Publish placement location
        self.place_pub = rospy.Publisher('/place_loc', Float32MultiArray, queue_size=10)
        
        rospy.spin()

def main():
    object_labeler = ObjectsLocator()
    object_labeler.run()

if __name__ == '__main__':
    main()