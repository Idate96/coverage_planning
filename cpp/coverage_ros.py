"""
This file contains the ros interface for the package coverage.
"""
import numpy as np
import rospy
import numpy
import std_msgs.msg
import geometry_msgs.msg


def coverage_ros():
    """
    This function is the ros interface for the package coverage.
    """
    rospy.init_node('coverage_node')
    # load numpy array from csv file
    path_name = rospy.get_param('path_name')
    path = np.load(path_name, allow_pickle=True)
    # convert numpy array into ros Poses
    poses = []
    for i in range(len(path)):
        pose = geometry_msgs.msg.Pose()
        pose.position.x = path[i, 0]
        pose.position.y = path[i, 1]
        pose.position.z = 0
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1
        poses.append(pose)

    print("loaded all poses")
    # publish ros Poses
    pub = rospy.Publisher('/coverage/poses', geometry_msgs.msg.PoseArray, queue_size=10)
    # create header
    header = std_msgs.msg
    header.frame_id = 'map'
    header.stamp = rospy.Time.now()

    # create PoseArray
    pose_array = geometry_msgs.msg.PoseArray(header, poses)
    while not rospy.is_shutdown():
        # count number of subscribers
        if pub.get_num_connections() > 0:
            pub.publish(pose_array)
            # shutdown the publisher
            rospy.signal_shutdown("shutdown")


if __name__ == "__main__":
    coverage_ros()