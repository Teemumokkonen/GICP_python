import numpy as np
import open3d as o3d
from GICP.registeration import GICP
def create_sample_point_clouds():
    # Generate a random target point cloud
    target_cloud = o3d.geometry.PointCloud()
    target_points = np.random.uniform(-1, 1, (200, 3))  # 100 random points in 3D space
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    # Generate a random source point cloud by applying a slight transformation to the target
    source_cloud = o3d.geometry.PointCloud()
    transformation = np.array([[ 0.9984775, -0.0523280, 0.0174524, 0.01],    
                               [0.0545673, 0.9832995, -0.1736217, 0],        # Translate 0 
                               [-0.0080757, 0.1743097, 0.9846578 , 0.01],     # Translate 0
                               [0, 0, 0, 1]])      

    source_points = np.dot(np.hstack((target_points, np.ones((200, 1)))), transformation.T)[:, :3]
    source_cloud.points = o3d.utility.Vector3dVector(source_points)

    return target_cloud, source_cloud

# Visualize the sample point clouds
def visualize_point_clouds(target, source):
    # Set different colors for target and source clouds for easier distinction
    target.paint_uniform_color([1, 0, 0])  # Red for target
    source.paint_uniform_color([0, 1, 0])  # Green for source

    # Visualize both clouds
    o3d.visualization.draw_geometries([target, source],
                                      window_name="Target and Source Point Clouds",
                                      width=800, height=600)

if __name__ == "__main__":
    # Create the target and source point clouds
    target_cloud, source_cloud = create_sample_point_clouds()
    transformation = np.array([[ 0.9984775, -0.0523280, 0.0174524, 0.01],    
                               [0.0545673, 0.9832995, -0.1736217, 0],        # Translate 0 
                               [-0.0080757, 0.1743097, 0.9846578 , 0.01],     # Translate 0
                               [0, 0, 0, 1]])      
   
    # Visualize the initial alignment of point clouds
    visualize_point_clouds(target_cloud, source_cloud)
    gicp = GICP()
    gicp.set_source(source_cloud)
    gicp.set_target(target_cloud)
    print(gicp.compute_transformation(np.eye(4,4)))
    print(transformation)

