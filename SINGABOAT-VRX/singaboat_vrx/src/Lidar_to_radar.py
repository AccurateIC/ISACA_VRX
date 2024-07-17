import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Int64, String, Float64
from sensor_msgs.msg import PointCloud2
from radar_custom_msg.msg import RadarArray, Radar  # Adjust according to your package structure
import rospy
from scipy import stats

class LidarToRadarConverter:
    def __init__(self, task_name="perception"):
        self.task_name = task_name
        self.pc_centroids = []
        self.rgb_objects = []
        self.detected_objects = []
        self.detected_obstacles = []
        self.time_slot = 0
        self.debug = True
        self.obstacle_msg = Float64()
        # self.obstacle_pub = rospy.Publisher('/obstacle', Float64, queue_size=10)
        self.radar_pub = rospy.Publisher('/radar_data', RadarArray, queue_size=10)

    def lidar_callback(self, msg):
        
        dbscan_min_points = 3
        ros_point_cloud = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        open3d_point_cloud = o3d.geometry.PointCloud()
        open3d_point_cloud_wide = o3d.geometry.PointCloud()

        if len(ros_point_cloud) != 0:
            xyz_filter = []
            xyz_filter_wide = []
            xyz = [(x, y, 0) for x, y, z in ros_point_cloud]
            for i in range(len(xyz)):
                if 1 < xyz[i][0] < 35 and abs(xyz[i][1]) < 20:
                    xyz_filter_wide.append(xyz[i])
                    if abs(xyz[i][1] / xyz[i][0]) < 1:
                        xyz_filter.append(xyz[i])
                        if xyz[i][0] < 10:
                            if abs(xyz[i][1] / xyz[i][0]) < 0.8:
                                xyz_filter.append(xyz[i])

            if len(xyz_filter_wide) != 0 and len(xyz_filter) != 0:
                xyz_array = np.array(xyz_filter)
                open3d_point_cloud.points = o3d.utility.Vector3dVector(xyz_array)
                xyz_array_wide = np.array(xyz_filter_wide)
                open3d_point_cloud_wide.points = o3d.utility.Vector3dVector(xyz_array_wide)

                pc_clusters = np.array(open3d_point_cloud.cluster_dbscan(eps=0.3, min_points=dbscan_min_points, print_progress=False))
                pc_clusters_wide = np.array(open3d_point_cloud_wide.cluster_dbscan(eps=0.3, min_points=dbscan_min_points, print_progress=False))
                max_pc_cluster = pc_clusters.max()
                max_pc_cluster_wide = pc_clusters_wide.max()

                pc_centroids_sum = np.zeros((max_pc_cluster + 1, 3))
                counters = np.zeros(max_pc_cluster + 1)
                pc_centroids = np.zeros((max_pc_cluster + 1, 3))
                for i in range(len(pc_clusters + 1)):
                    if pc_clusters[i] >= 0:
                        pc_centroids_sum[pc_clusters[i]] += xyz_array[i]
                        counters[pc_clusters[i]] += 1
                for i in range(max_pc_cluster + 1):
                    pc_centroids[i] = pc_centroids_sum[i] / counters[i]
                pc_centroids_ranked = self.rank_pc_centroids(pc_centroids)

                pc_centroids_sum_wide = np.zeros((max_pc_cluster_wide + 1, 3))
                counters_wide = np.zeros(max_pc_cluster_wide + 1)
                pc_centroids_wide = np.zeros((max_pc_cluster_wide + 1, 3))
                for i in range(len(pc_clusters_wide + 1)):
                    if pc_clusters_wide[i] >= 0:
                        pc_centroids_sum_wide[pc_clusters_wide[i]] += xyz_array_wide[i]
                        counters_wide[pc_clusters_wide[i]] += 1
                for i in range(max_pc_cluster_wide + 1):
                    pc_centroids_wide[i] = pc_centroids_sum_wide[i] / counters_wide[i]
                pc_centroids_ranked_wide = self.rank_pc_centroids(pc_centroids_wide)

                if self.task_name == "perception":
                    if len(pc_centroids_ranked) > len(self.pc_centroids):
                        self.pc_centroids = pc_centroids_ranked
                    if self.time_slot < 5.5:
                        self.pc_centroids = []
                    if 8.9 < self.time_slot < 9.8:
                        if len(self.rgb_objects) != 0 and len(self.pc_centroids) != 0:
                            rgb_objects_ranked = self.rank_rgb_objects(self.rgb_objects)
                            self.detected_objects = self.fuse_camera_lidar_data(self.pc_centroids, rgb_objects_ranked)
                else:
                    self.pc_centroids = pc_centroids_ranked
                    if len(self.rgb_objects) != 0 and len(self.pc_centroids) != 0:
                        rgb_objects_ranked = self.rank_rgb_objects(self.rgb_objects)
                        self.detected_objects = self.fuse_camera_lidar_data(self.pc_centroids, rgb_objects_ranked)
                    self.detected_obstacles = []
                    if len(pc_centroids_wide) != 0:
                        for i in range(len(pc_centroids_wide)):
                            self.detected_obstacles.append((pc_centroids_wide[i, 0:2]).tolist())
                    self.obstacle_msg.data = sum(self.detected_obstacles, [])
                    self.obstacle_pub.publish(self.obstacle_msg)

                self.publish_radar_data(pc_centroids_ranked)

    def publish_radar_data(self, pc_centroids_ranked):
        radar_array_msg = RadarArray()
        radar_array_msg.header = Header()
        radar_array_msg.header.stamp = rospy.Time.now()
        radar_array_msg.header.frame_id = "radar_frame"

        radar_targets = []
        for i, centroid in enumerate(pc_centroids_ranked):
            radar_msg = Radar()
            radar_msg.header = Header()
            radar_msg.header.stamp = rospy.Time.now()
            radar_msg.header.frame_id = "radar_target_frame"
            radar_msg.target_id = Int64(data=i)
            radar_msg.valid = String(data="true")
            radar_msg.autoaquire = String(data="true")
            radar_msg.state = String(data="detected")
            radar_msg.range = Float64(data=np.linalg.norm(centroid))
            radar_msg.bearing = Float64(data=np.arctan2(centroid[1], centroid[0]))
            radar_msg.true_bearing = Float64(data=np.arctan2(centroid[1], centroid[0]))
            radar_msg.relative_speed = Float64(data=0.0)  # Assuming stationary for simplicity
            radar_msg.relative_course = Float64(data=0.0)  # Assuming stationary for simplicity
            radar_msg.true_speed = Float64(data=0.0)  # Assuming stationary for simplicity
            radar_msg.true_course = Float64(data=0.0)  # Assuming stationary for simplicity
            radar_msg.cpa = Float64(data=0.0)  # Assuming CPA is not computed
            radar_msg.tcp = Int64(data=0)  # Assuming TCP is not computed
            radar_msg.headinga_at_last_update = Float64(data=0.0)  # Assuming heading is not computed
            radar_msg.going_to_cp = String(data="false")
            radar_msg.true_data_valid = String(data="true")

            radar_targets.append(radar_msg)

        radar_array_msg.targets = radar_targets
        self.radar_pub.publish(radar_array_msg)

    def rank_rgb_objects(self, rgb_objects):
        rgb_objects_pos = []
        for i in range(len(rgb_objects)):
            if (i + 1) % 2 == 0:
                rgb_objects_pos.append(rgb_objects[i])
        x = stats.rankdata(np.array(rgb_objects_pos)[:, 0], method='dense')
        x_list = x.tolist()
        rgb_objects_ranked = []
        for i in range(int(len(rgb_objects) / 2)):
            index = x_list.index(i + 1)
            rgb_objects_ranked.append(rgb_objects[2 * index])
            rgb_objects_ranked.append(rgb_objects[2 * index + 1])
        return rgb_objects_ranked

    def rank_pc_centroids(self, pc_centroids):
        slope = pc_centroids[:, 1] / pc_centroids[:, 0]
        y = stats.rankdata(slope, method='dense')
        y = len(y) + 1 - y.astype(int)
        y_list = y.tolist()
        pc_centroids_ranked = []
        for i in range(len(pc_centroids)):
            pc_centroids_ranked.append(pc_centroids[y_list.index(i + 1)])
        pc_centroids_ranked = np.array(pc_centroids_ranked)
        return pc_centroids_ranked

    def fuse_camera_lidar_data(self, pc_centroids_ranked, rgb_objects_ranked):
        detected_objects = []
        if len(rgb_objects_ranked) / 2 == len(pc_centroids_ranked):
            for i in range(len(pc_centroids_ranked)):
                detected_objects.append(rgb_objects_ranked[2 * i])
                detected_objects.append(rgb_objects_ranked[2 * i + 1])
                detected_objects.append(pc_centroids_ranked[i])
        return detected_objects


        
if __name__ == "__main__":
    converter = LidarToRadarConverter(task_name="perception")
    rospy.init_node('lidar_to_radar_converter', anonymous=True)
    rospy.Subscriber('/usv/sensors/lidars_points', PointCloud2, converter.lidar_callback)
    rospy.spin()
