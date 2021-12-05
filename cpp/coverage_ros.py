"""
This file contains the ros interface for the package coverage.
"""
import numpy as np
import rospy
import tf
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
    print(path)
    for i in range(len(path)):
        pose = geometry_msgs.msg.Pose()
        pose.position.x = path[i, 0]
        pose.position.y = path[i, 1]
        pose.position.z = 0
        # transform euler angles to quaternions
        quaternion = tf.transformations.quaternion_from_euler(0, 0, path[i, 2])
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
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
            # print(pose_array)
            # shutdown the publisher
            rospy.signal_shutdown("shutdown")


if __name__ == "__main__":
    coverage_ros()