import spark_dsg as dsg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gtsam
from gtsam import Pose3, Rot3, Point3, symbol, NonlinearFactorGraph, Values, noiseModel
import os
import open3d as o3d
from collections import defaultdict
from scipy.linalg import expm
import copy

np.random.seed(14)

# =============================================================================
# Load the Scene Graph and Remove Nodes
# =============================================================================
data_folder = "scene_graphs"

# filename = "example_dsg_for_jared.json"
# filename = "spot_building45_gadget.json"
filename = "apartment_dsg.json"

path_to_dsg = os.path.join(data_folder, filename)

G = dsg.DynamicSceneGraph.load(path_to_dsg)

# ########## Remove Some Object Nodes
object_layer = G.get_layer(dsg.DsgLayers.OBJECTS)
object_nodes = list(object_layer.nodes)
nodes_to_remove = object_nodes[:2] + object_nodes[-4:]
for obj in nodes_to_remove:
    G.remove_node(obj.id.value)

# =============================================================================
# Extract Original (Prior) Data
# =============================================================================
# Extract Agent Data
agent_layer = G.get_dynamic_layer(dsg.DsgLayers.AGENTS, "a")
agent_trajectories = []
for agent in agent_layer.nodes:
    timestamp = agent.timestamp
    pos = agent.attributes.position if hasattr(agent.attributes, "position") else None
    R_body = agent.attributes.world_R_body
    if pos is not None:
        pos = np.array(pos).flatten()
    agent_trajectories.append({
        "id": agent.id,
        "timestamp": timestamp,
        "position": pos,
        "orientation": R_body
    })

# Extract Object Data
objects_data = []
object_layer = G.get_layer(dsg.DsgLayers.OBJECTS)
for obj in object_layer.nodes:
    pos = obj.attributes.position if hasattr(obj.attributes, "position") else None
    R_object = obj.attributes.world_R_object if hasattr(obj.attributes, "world_R_object") else None
    bounding_box = obj.attributes.bounding_box if hasattr(obj.attributes, "bounding_box") else None
    semantic_label = obj.attributes.semantic_label if hasattr(obj.attributes, "semantic_label") else None
    semantic_feature = obj.attributes.semantic_feature if hasattr(obj.attributes, "semantic_feature") else None
    if pos is not None:
        pos = np.array(pos).flatten()
    if bounding_box is not None:
        bbox = {}
        bbox["pos"] = np.array(bounding_box.world_P_center).flatten()
        bbox["dim"] = np.array(bounding_box.dimensions).flatten()
        bbox["min"] = np.array(bounding_box.min).flatten()
        bbox["max"] = np.array(bounding_box.max).flatten()
    objects_data.append({
        "id": obj.id,
        "position": pos,
        "orientation": R_object,
        "bounding_box": bbox,
        "semantic_label": semantic_label,
        "semantic_feature": semantic_feature,
    })

# Generate point clouds from bounding boxes
def generate_point_cloud_from_bbox(bbox, n_points=500):
    if bbox is None:
        return None

    bb_min = bbox["min"]
    bb_max = bbox["max"]
    center = (bb_min + bb_max) / 2.0
    dims = bb_max - bb_min

    # Create a box mesh with the given dimensions.
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=dims[0], height=dims[1], depth=dims[2])
    mesh_box.compute_vertex_normals()
    # By default, create_box places the lower corner at the origin (so center is at dims/2).
    # To position the box so that its center is at 'center', translate by center - dims/2.
    translation = center - dims / 2.0
    mesh_box.translate(translation)

    # Sample points uniformly from the mesh surface for a denser, mesh-like point cloud.
    pcd = mesh_box.sample_points_uniformly(number_of_points=n_points)
    return pcd

def generate_point_cloud_from_bbox_by_resolution(bbox, resolution=0.01):
    if bbox is None:
        return None

    bb_min = bbox["min"]
    bb_max = bbox["max"]

    # For each axis, determine a grid using linspace.
    def grid_points(min_val, max_val, res):
        # Number of points = max(2, ceil((max - min) / res) + 1)
        n = max(2, int(np.ceil((max_val - min_val) / res)) + 1)
        return np.linspace(min_val, max_val, n)

    xs = grid_points(bb_min[0], bb_max[0], resolution)
    ys = grid_points(bb_min[1], bb_max[1], resolution)
    zs = grid_points(bb_min[2], bb_max[2], resolution)

    # Create grids for each face.
    # Face 1: z = bb_min[2], x and y vary.
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    face1 = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, bb_min[2])))
    
    # Face 2: z = bb_max[2], x and y vary.
    face2 = np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, bb_max[2])))
    
    # Face 3: y = bb_min[1], x and z vary.
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    face3 = np.column_stack((X.ravel(), np.full(X.size, bb_min[1]), Z.ravel()))
    
    # Face 4: y = bb_max[1], x and z vary.
    face4 = np.column_stack((X.ravel(), np.full(X.size, bb_max[1]), Z.ravel()))
    
    # Face 5: x = bb_min[0], y and z vary.
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    face5 = np.column_stack((np.full(Y.size, bb_min[0]), Y.ravel(), Z.ravel()))
    
    # Face 6: x = bb_max[0], y and z vary.
    face6 = np.column_stack((np.full(Y.size, bb_max[0]), Y.ravel(), Z.ravel()))
    
    # Combine all faces and remove duplicates.
    all_points = np.vstack((face1, face2, face3, face4, face5, face6))
    unique_points = np.unique(all_points, axis=0)
    
    # Create an Open3D point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)
    return pcd
    
def load_bunny_pcd(bbox):
    # Load the Stanford Bunny mesh from Open3D's built-in data.
    bunny_data = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny_data.path)
    mesh.compute_vertex_normals()

    # Sample points from the mesh.
    bunny_pcd = mesh.sample_points_uniformly(number_of_points=3000)
    
    # Compute the bunny's current axis-aligned bounding box.
    bunny_bbox = bunny_pcd.get_axis_aligned_bounding_box()
    bunny_center = bunny_bbox.get_center()
    bunny_dims = bunny_bbox.get_max_bound() - bunny_bbox.get_min_bound()
    
    # Extract the target bounding box from the provided bbox.
    target_bb_min = bbox["min"]
    target_bb_max = bbox["max"]
    target_center = (target_bb_min + target_bb_max) / 2.0
    target_dims = target_bb_max - target_bb_min
    
    # Compute scale factor so bunny fits inside the target bounding box.
    scale_factors = target_dims / bunny_dims
    scale = np.min(scale_factors)
    
    # Scale bunny with respect to its center.
    bunny_pcd.scale(scale, center=bunny_center)
    
    # Translate bunny's center to the target bounding box center.
    bunny_pcd.translate(target_center - bunny_center)
    
    return bunny_pcd

# Example: generate point clouds for all objects with bounding boxes in your objects_data list.
for obj in objects_data:
    bbox = obj.get("bounding_box", None)
    # pcd = generate_point_cloud_from_bbox(bbox, n_points=1000)
    # pcd = generate_point_cloud_from_bbox_by_resolution(bbox, resolution=0.1)
    pcd = load_bunny_pcd(bbox)
    obj["pcd"] = pcd
    if pcd is not None:
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        obj["centroid"] = centroid
        # print(f"Object {obj['id']}: position={obj['position']}, centroid={centroid}")
        # o3d.visualization.draw_geometries([pcd])

# =============================================================================
# Create Measurement Edges between Original (Prior) Agents and Object Centroids
# =============================================================================
radius_threshold = 6
gt_measurement_edges = []
for obj in objects_data:
    obj_centroid = obj["centroid"]
    if obj_centroid is None:
        continue
    for agent in agent_trajectories:
        agent_pos = agent["position"]
        if agent_pos is None:
            continue
        distance = np.linalg.norm(obj_centroid - agent_pos)
        if distance <= radius_threshold:
            R_agent = gtsam.Rot3.Quaternion(
                agent["orientation"].w,
                agent["orientation"].x,
                agent["orientation"].y,
                agent["orientation"].z).matrix()
    
            edge = {
                "object_id": obj["id"],
                "agent_id": agent["id"],
                "distance": distance,
                "object_position": obj_centroid,
                "agent_position": agent_pos,
                "agent_orientation": agent["orientation"],
                "agent_orientation_matrix": R_agent,
                "object_orientation": obj["orientation"],
            }
            gt_measurement_edges.append(edge)

# =============================================================================
# Define Helper Functions to Add Noise and Simulate Cumulative Drift
# =============================================================================

def randlangevin(mode, kappa):
    """
    Sample from the Langevin distribution in SO(3) with a given mode and concentration parameter.
    
    Parameters:
    -----------
    mode : array_like, shape (3, 3)
        The mode rotation matrix.
    kappa : float
        The concentration parameter. If kappa <= 0, the function returns the identity matrix.
        
    Returns:
    --------
    Re : ndarray, shape (3, 3)
        A random rotation matrix sampled from the Langevin distribution.
    """
    if kappa <= 0:
        return np.eye(3)
    
    # 1) Sample theta from the von Mises distribution
    theta = np.random.vonmises(0, 2 * kappa)
    
    # 2) Sample an axis of rotation uniformly from the sphere.
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    
    # 3) Construct the skew-symmetric matrix corresponding to the rotation axis.
    axis_skew = np.array([[0,         -axis[2],  axis[1]],
                          [axis[2],    0,       -axis[0]],
                          [-axis[1],   axis[0],  0]])
    
    # Compute the rotation matrix P via the matrix exponential.
    P = expm(theta * axis_skew)
    
    # Multiply the mode by the perturbation P.
    Re = mode @ P
    return Re

def add_cumulative_drift_to_agents(agent_list, drift_std=0.05, alpha=0.9):
    noisy_agents = []
    drift_offset = np.zeros(3)
    previous_increment = np.zeros(3)
    sorted_agents = sorted(agent_list, key=lambda x: x["timestamp"])
    for agent in sorted_agents:
        new_increment = alpha * previous_increment + (1 - alpha) * np.random.normal(0, drift_std, 3)
        drift_offset += new_increment
        previous_increment = new_increment
        noisy_agent = agent.copy()
        noisy_agent["position"] = agent["position"] + drift_offset
        noisy_agents.append(noisy_agent)
    return noisy_agents

def add_orientation_langevin_noise_to_agents(agent_list, kappa):
    noisy_agents = []
    for agent in agent_list:
        noisy_agent = agent.copy()
        # Generate Langevin noise for the orientation.
        mode = agent["orientation"] # Quaternion
        mode = gtsam.Rot3.Quaternion(mode.w, mode.x, mode.y, mode.z).matrix()
        noisy_orientation = randlangevin(mode, kappa)
        noisy_agent["orientation_matrix"] = noisy_orientation # Matrix 
        noisy_agents.append(noisy_agent)

        viz = False
        if viz:
            # Create a 3D plot to visualize the camera frames.
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Define origins and colors for each axis
            origin = [0, 0, 0]
            colors = ['r', 'g', 'b']

            # Plot the original (mode) coordinate frame with solid lines
            for i, color in enumerate(colors):
                ax.quiver(origin[0], origin[1], origin[2],
                        mode[0, i], mode[1, i], mode[2, i],
                        color=color, arrow_length_ratio=0.1, label=f"Original axis {i+1}")

            # Plot the noisy (noisy_orientation) coordinate frame with dashed lines
            for i, color in enumerate(colors):
                ax.quiver(origin[0], origin[1], origin[2],
                        noisy_orientation[0, i], noisy_orientation[1, i], noisy_orientation[2, i],
                        color=color, linestyle='dashed', arrow_length_ratio=0.1, label=f"Noisy axis {i+1}")

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Original (solid) vs Noisy (dashed) Rotation Frames")
            ax.legend()
            plt.show()
    return noisy_agents

def generate_measurement_noise_perturbation(trans_std=0.1, rot_std=0.05):
    # Generate random noise for translation (3D vector)
    trans_noise = np.random.normal(0, trans_std, 3)

    # Generate random noise for rotation.
    # Create a small rotation using the exponential map from a rotation vector.
    rot_noise_vec = np.random.normal(0, rot_std, 3)
    rot_noise = gtsam.Rot3.Expmap(rot_noise_vec)

    # Create a noise pose from the rotation and translation noise.
    noise_pose = Pose3(rot_noise, Point3(trans_noise[0], trans_noise[1], trans_noise[2]))
    return noise_pose

# =============================================================================
# Create Noisy Copies and Measurement Edges for Noisy Data
# =============================================================================
noisy_agent_trajectories = add_cumulative_drift_to_agents(agent_trajectories, drift_std=0.05, alpha=0.9)
noisy_agent_trajectories = add_orientation_langevin_noise_to_agents(noisy_agent_trajectories, kappa=50.0)

# =============================================================================
# Place Noisy Objects in the Scene, Hydra Data ("Hydra Frame")
# =============================================================================
noisy_objects_data = []

grouped_edges = defaultdict(list)
for edge in gt_measurement_edges:
    grouped_edges[edge["object_id"]].append(edge)

# For each unique object, select the middle edge and compute the relative transform.
measurement_edges = []
for obj_id, edges in grouped_edges.items():
    mid_index = len(edges) // 2
    mid_edge = edges[mid_index]
    
    # Find the corresponding agent and object
    agent = next((a for a in agent_trajectories if a["id"] == mid_edge["agent_id"]), None)
    obj   = next((o for o in objects_data if o["id"] == mid_edge["object_id"]), None)
    if agent is None or obj is None:
        continue

    noisy_agent = next((a for a in noisy_agent_trajectories if a["id"] == mid_edge["agent_id"]), None)
    noisy_object_centroid = noisy_agent["position"] + (obj["centroid"] - agent["position"])
    original_to_noisy = noisy_object_centroid - obj["centroid"]

    def add_noise_to_point_cloud(pcd, noise_std=0.005):
        points = np.asarray(pcd.points)
        noise = np.random.normal(0, noise_std, points.shape)
        noisy_points = points + noise
        pcd.points = o3d.utility.Vector3dVector(noisy_points)
        return pcd

    pcd_copy = copy.deepcopy(obj["pcd"])
    points = np.asarray(pcd_copy.points)
    centroid = np.mean(points, axis=0)
    # Slice the point cloud: keep points with x-coordinate less than the centroid's x (one half)
    mask = points[:, 0] < centroid[0]
    filtered_points = points[mask]
    pcd_copy.points = o3d.utility.Vector3dVector(filtered_points)
    shifted_pcd = pcd_copy.translate(original_to_noisy)
    noisy_pcd = add_noise_to_point_cloud(shifted_pcd, noise_std=0.01)
    new_centroid = np.mean(np.asarray(noisy_pcd.points), axis=0)

    # Convert the original agent orientation from mid_edge to a gtsam.Rot3.
    R_agent_orig = gtsam.Rot3(mid_edge["agent_orientation_matrix"])
    R_agent_noisy = gtsam.Rot3(noisy_agent["orientation_matrix"])

    # Compute the relative rotation between the noisy and original agent orientations.
    R_relative = R_agent_noisy.compose(R_agent_orig.inverse())

    # Convert the mid_edge object orientation into a gtsam.Rot3.
    R_obj_orig = gtsam.Rot3.Quaternion(
        mid_edge["object_orientation"].w,
        mid_edge["object_orientation"].x,
        mid_edge["object_orientation"].y,
        mid_edge["object_orientation"].z)
    
    # Apply the relative rotation to the object orientation.
    R_obj_noisy = R_relative.compose(R_obj_orig).matrix()

    # Translate the bounding box if available.
    translated_bbox = {}
    if obj["bounding_box"] is not None:
        translated_bbox["pos"] = bbox["pos"] + original_to_noisy
        translated_bbox["min"] = bbox["min"] + original_to_noisy
        translated_bbox["max"] = bbox["max"] + original_to_noisy

    noisy_objects_data.append({
        "id": obj["id"],
        "centroid": new_centroid,
        "orientation": R_obj_noisy,
        "bounding_box": translated_bbox,
        "pcd": noisy_pcd,
    })

    measurement_edges.append({
        "object_id": obj["id"],
        "agent_id": noisy_agent["id"],
        "distance": mid_edge["distance"],
        "object_position": new_centroid,
        "agent_position": noisy_agent["position"],
    })

# =============================================================================
# Perform ICP Registration to Align Noisy Objects with Prior Objects
# =============================================================================
def register_icp(source_pcd, target_pcd, threshold=0.5, trans_init=np.eye(4)):
    reg = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg

from pycpd import RigidRegistration
def register_cpd(source_pcd, target_pcd):
    # Convert Open3D point clouds to numpy arrays.
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    # Create and run CPD registration.
    reg = RigidRegistration(X=target_points, Y=source_points)
    TY, (s_reg, R_reg, t_reg) = reg.register()
    
    # Build a 4x4 transformation matrix from the CPD registration parameters.
    T = np.eye(4)
    T[:3, :3] = s_reg * R_reg  # Apply the scale and rotation.
    T[:3, 3] = t_reg           # Set the translation.
    
    return T, TY

registration_method = "icp"  # Choose between "icp" and "cpd"

# For each noisy object, find the corresponding original object (by id) and perform ICP.
for noisy_obj in noisy_objects_data:
    # Look up the corresponding original object (prior) by object id.
    original_obj = next((o for o in objects_data if o["id"] == noisy_obj["id"]), None)
    if original_obj is None:
        continue
    if "pcd" not in original_obj or "pcd" not in noisy_obj:
        continue

    source = copy.deepcopy(noisy_obj["pcd"])  # Observed (noisy) point cloud.
    target = original_obj["pcd"]              # Prior point cloud.

    source.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    target.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

    # --- Align centroids first ---
    # Compute the offset between the centroids.
    centroid_offset = np.array(original_obj["centroid"]) - np.array(noisy_obj["centroid"])
    print(f"Aligning centroids for object {noisy_obj['id']} with offset: {centroid_offset}")
    # Translate the source point cloud by the centroid offset.
    source.translate(centroid_offset)

    # o3d.visualization.draw_geometries([source, target])
    
    if registration_method == "icp":
        # --- Perform ICP ---
        # Set an ICP threshold (adjust if needed)
        threshold = 1.0
        # Use identity since we already aligned the centroids.
        trans_init = np.eye(4)
        reg_result = register_icp(source, target, threshold, trans_init)

        print(f"ICP for object {noisy_obj['id']} transformation:")
        print(reg_result.transformation)

        # Store the transformation.
        noisy_obj["registration_transform"] = reg_result.transformation

        # Apply the transformation to the source point cloud.
        source.transform(reg_result.transformation)

    elif registration_method == "cpd":
        # --- Perform CPD ---
        T, TY = register_cpd(source, target)
        print(f"CPD for object {noisy_obj['id']} transformation:")
        print(T)

        # Store the transformation.
        noisy_obj["registration_transform"] = T

        # Apply the transformation to the source point cloud.
        source.transform(T)

    # Visualize the aligned point clouds.
    # o3d.visualization.draw_geometries([source, target])

    # Find centroid pose difference between original and noisy object after alignment
    source_centroid = np.mean(np.asarray(source.points), axis=0)
    target_centroid = np.mean(np.asarray(target.points), axis=0)
    centroid_offset = target_centroid - source_centroid

    # Construct measurement edge
    # There is only one measurement edge per object
    measurement_edge = next((e for e in measurement_edges if e["object_id"] == noisy_obj["id"]), None)
    if measurement_edge is not None:
        measurement_edge["centroid_offset"] = centroid_offset
        measurement_edge["agent_to_prior_centroid_translation"] = measurement_edge["object_position"] - measurement_edge["agent_position"] + centroid_offset


# =============================================================================
# Build a GTSAM Factor Graph
# =============================================================================
graph = NonlinearFactorGraph()
initial_estimates = Values()

# sorted_agents = sorted(agent_trajectories, key=lambda x: x["timestamp"])
n_agents = len(agent_trajectories)
n_landmarks = len(objects_data)

# Create initial estimates for agent poses.
agent_poses = []
for i, agent in enumerate(noisy_agent_trajectories):
    q = agent["orientation"]
    R = Rot3.Quaternion(q.w, q.x, q.y, q.z)
    t = Point3(agent["position"][0], agent["position"][1], agent["position"][2])
    pose = Pose3(R, t)
    agent_poses.append(pose)
    key = symbol('x', i)
    initial_estimates.insert(key, pose)

# Sanity check using the original agent poses.
# agent_poses = []
# for i, agent in enumerate(agent_trajectories):
#     q = agent["orientation"]
#     R = Rot3.Quaternion(q.w, q.x, q.y, q.z)
#     t = Point3(agent["position"][0], agent["position"][1], agent["position"][2])
#     pose = Pose3(R, t)
#     agent_poses.append(pose)
#     key = symbol('x', i)
#     initial_estimates.insert(key, pose)

# Create initial estimates for landmarks (using original objects_data priors).
landmark_poses = []
for j, obj in enumerate(objects_data):
    t = Point3(obj["position"][0], obj["position"][1], obj["position"][2])
    landmark_pose = Pose3(Rot3(), t)  # Identity rotation.
    landmark_poses.append(landmark_pose)
    key = symbol('l', j)
    initial_estimates.insert(key, landmark_pose)

# Add odometry factors between consecutive agent poses.
odometry_noise = noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
for i in range(n_agents - 1):
    key1 = symbol('x', i)
    key2 = symbol('x', i+1)
    odometry = agent_poses[i].between(agent_poses[i+1])
    graph.add(gtsam.BetweenFactorPose3(key1, key2, odometry, odometry_noise))

# Add relative pose factors from agents to landmarks.
relative_noise = noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
for edge in measurement_edges:
    agent_index = next((i for i, a in enumerate(noisy_agent_trajectories) if a["id"] == edge["agent_id"]), None)
    # Find the corresponding landmark in the noisy objects.
    landmark_index = next((j for j, o in enumerate(noisy_objects_data) if o["id"] == edge["object_id"]), None)
    if agent_index is None or landmark_index is None:
        continue
    key_agent = symbol('x', agent_index)
    key_landmark = symbol('l', landmark_index)
    # Construct Pose3 for the noisy agent.
    agent = agent_trajectories[agent_index]
    q_a = agent["orientation"]
    agent_pose = Pose3(Rot3.Quaternion(q_a.w, q_a.x, q_a.y, q_a.z),
                             Point3(agent["position"][0],
                                    agent["position"][1],
                                    agent["position"][2]))
    object = next((o for o in objects_data if o["id"] == edge["object_id"]), None)
    if object is None:
        continue

    # Construct Pose3 for the landmark.
    q_o = object["orientation"]
    landmark_pose = Pose3(Rot3.Quaternion(q_o.w, q_o.x, q_o.y, q_o.z), 
                                        Point3(object["centroid"][0] + edge["centroid_offset"][0],
                                               object["centroid"][1] + edge["centroid_offset"][1],
                                               object["centroid"][2] + edge["centroid_offset"][2]))
    # Compute the relative measurement: T_agent^{-1} * T_landmark.
    relative_measurement = agent_pose.between(landmark_pose)

    # Add noise to the relative measurement.
    # noise_pose = generate_measurement_noise_perturbation(trans_std=0.01, rot_std=0.02)
    # noisy_relative_measurement = relative_measurement.compose(noise_pose)

    # Add the factor to the graph.
    # graph.add(gtsam.BetweenFactorPose3(key_agent, key_landmark, noisy_relative_measurement, relative_noise))
    graph.add(gtsam.BetweenFactorPose3(key_agent, key_landmark, relative_measurement, relative_noise))

# Add prior factors for landmarks using the original objects_data.
prior_noise = noiseModel.Isotropic.Sigma(6, 0.1)  # 6-dim noise.
for j, obj in enumerate(objects_data):
    key = symbol('l', j)
    prior = Pose3(Rot3(), Point3(obj["centroid"][0], obj["centroid"][1], obj["centroid"][2]))
    graph.add(gtsam.PriorFactorPose3(key, prior, prior_noise))

# =============================================================================
# Optimize the Factor Graph
# =============================================================================
params = gtsam.LevenbergMarquardtParams()
params.setVerbosityLM("SUMMARY")
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
result = optimizer.optimize()

# Retrieve optimized agent poses and landmark positions.
optimized_agent_poses = [result.atPose3(symbol('x', i)) for i in range(n_agents)]
optimized_landmark_positions = [result.atPose3(symbol('l', j)).translation() for j in range(n_landmarks)]

print("Optimized Agent Poses:")
for i, pose in enumerate(optimized_agent_poses):
    print(f"Agent {i}: {pose}")

print("\nOptimized Landmark Positions:")
for j, point in enumerate(optimized_landmark_positions):
    print(f"Landmark {j}: {point}")

# =============================================================================
# Visualization of Priors and Optimized Results
# =============================================================================

agent_x = [a["position"][0] for a in agent_trajectories if a["position"] is not None]
agent_y = [a["position"][1] for a in agent_trajectories if a["position"] is not None]
agent_z = [a["position"][2] for a in agent_trajectories if a["position"] is not None]

object_x = [o["position"][0] for o in objects_data if o["position"] is not None]
object_y = [o["position"][1] for o in objects_data if o["position"] is not None]
object_z = [o["position"][2] for o in objects_data if o["position"] is not None]

object_centroid_x = [o["centroid"][0] for o in objects_data if o["centroid"] is not None]
object_centroid_y = [o["centroid"][1] for o in objects_data if o["centroid"] is not None]
object_centroid_z = [o["centroid"][2] for o in objects_data if o["centroid"] is not None]

noisy_object_centroid_x = [o["centroid"][0] for o in noisy_objects_data if o["centroid"] is not None]
noisy_object_centroid_y = [o["centroid"][1] for o in noisy_objects_data if o["centroid"] is not None]
noisy_object_centroid_z = [o["centroid"][2] for o in noisy_objects_data if o["centroid"] is not None]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original agent trajectory (red) and object positions (blue)
ax.plot(agent_x, agent_y, agent_z, marker='o', color='red', label='Original Agent Trajectory')
ax.scatter(object_x, object_y, object_z, marker='s', color='blue', s=80, label='Original Object Position')
ax.scatter(object_centroid_x, object_centroid_y, object_centroid_z, marker='s', color='green', s=80, label='Original Object Centroid')
ax.scatter(noisy_object_centroid_x, noisy_object_centroid_y, noisy_object_centroid_z, marker='s', color='lime', s=80, label='Noisy Object Centroid')

# Plot original measurement edges as purple dashed lines.
for edge in measurement_edges:
    p_obj = edge["object_position"]
    p_agent = edge["agent_position"]
    ax.plot([p_obj[0], p_agent[0]],
            [p_obj[1], p_agent[1]],
            [p_obj[2], p_agent[2]],
            color='purple', linestyle='--', linewidth=1)
    # Line from agent to prior centroid (as if observed pcd was whole)
    p_prior = p_agent + edge["agent_to_prior_centroid_translation"]
    ax.plot([p_prior[0], p_agent[0]],
            [p_prior[1], p_agent[1]],
            [p_prior[2], p_agent[2]],
            color='red', linestyle='--', linewidth=1)

# Plot noisy agent positions (cyan) from cumulative drift.
noisy_agent_x = [a["position"][0] for a in noisy_agent_trajectories if a["position"] is not None]
noisy_agent_y = [a["position"][1] for a in noisy_agent_trajectories if a["position"] is not None]
noisy_agent_z = [a["position"][2] for a in noisy_agent_trajectories if a["position"] is not None]
ax.plot(noisy_agent_x, noisy_agent_y, noisy_agent_z, marker='o', color='cyan', label='Noisy Agent (Drifted)')

# Draw bounding boxes for original objects
for obj in objects_data:
    bb = obj.get("bounding_box", None)
    if bb is not None:
        try:
            bb_min = bb["min"]
            bb_max = bb["max"]
            if bb_min.size < 3 or bb_max.size < 3:
                continue
            x_min, y_min, z_min = bb_min[:3]
            x_max, y_max, z_max = bb_max[:3]
            vertices = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ])
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[3], vertices[0], vertices[4], vertices[7]]
            ]
            poly3d = Poly3DCollection(faces, edgecolors='green', facecolors=(0, 0, 0, 0), linewidths=1)
            ax.add_collection3d(poly3d)
        except Exception as e:
            print("Failed to add bounding box:", e)

# Draw bounding boxes for noisy objects
for obj in noisy_objects_data:
    bb = obj.get("bounding_box", None)
    if bb is not None:
        try:
            bb_min = bb["min"]
            bb_max = bb["max"]
            if bb_min.size < 3 or bb_max.size < 3:
                continue
            x_min, y_min, z_min = bb_min[:3]
            x_max, y_max, z_max = bb_max[:3]
            vertices = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ])
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[3], vertices[0], vertices[4], vertices[7]]
            ]
            poly3d = Poly3DCollection(faces, edgecolors='orange', facecolors=(0, 0, 0, 0), linewidths=1)
            ax.add_collection3d(poly3d)
        except Exception as e:
            print("Failed to add bounding box:", e)

# Overlay the prior point clouds
for obj in objects_data:
    if "pcd" in obj and obj["pcd"] is not None:
        pcd_points = np.asarray(obj["pcd"].points)
        ax.scatter(pcd_points[:,0], pcd_points[:,1], pcd_points[:,2],
                   s=2, color='black', alpha=0.5, label='Prior Point Cloud')

# Overlay the noisy point clouds
for obj in noisy_objects_data:
    if "pcd" in obj and obj["pcd"] is not None:
        pcd_points = np.asarray(obj["pcd"].points)
        ax.scatter(pcd_points[:,0], pcd_points[:,1], pcd_points[:,2],
                   s=2, color='magenta', alpha=0.5, label='Observed Point Cloud')

# Set equal axis scaling.
x_limits = ax.get_xlim3d()
y_limits = ax.get_ylim3d()
z_limits = ax.get_zlim3d()
x_range = abs(x_limits[1] - x_limits[0])
y_range = abs(y_limits[1] - y_limits[0])
z_range = abs(z_limits[1] - z_limits[0])
max_range = max(x_range, y_range, z_range)
x_mid = np.mean(x_limits)
y_mid = np.mean(y_limits)
z_mid = np.mean(z_limits)
ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Priors & Noisy Measurements: 3D Agent Trajectories and Object Positions")
ax.legend()
ax.grid(True)
plt.show()

# =======================================================================
# Additional Plot: Trajectories with Orientation at Agent Points
# =======================================================================

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Helper function: Given a quaternion (assumed to be an object with attributes w, x, y, z), return its rotation matrix.
def quaternion_to_matrix(q):
    return gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z).matrix()

# Define a scale factor for the orientation arrows.
arrow_scale = 0.5

# Sort agents by timestamp (for continuous trajectory plotting)
sorted_orig = sorted(agent_trajectories, key=lambda a: a["timestamp"])
sorted_noisy = sorted(noisy_agent_trajectories, key=lambda a: a["timestamp"])

# Extract trajectory coordinates.
orig_traj = np.array([agent["position"] for agent in sorted_orig if agent["position"] is not None])
noisy_traj = np.array([agent["position"] for agent in sorted_noisy if agent["position"] is not None])

# Plot trajectories as lines.
ax2.plot(orig_traj[:, 0], orig_traj[:, 1], orig_traj[:, 2],
         marker='o', color='red', label='Original Agent Trajectory')
ax2.plot(noisy_traj[:, 0], noisy_traj[:, 1], noisy_traj[:, 2],
         marker='o', color='cyan', label='Noisy Agent Trajectory')

# For each original agent, draw a quiver for the orientation.
for agent in sorted_orig:
    pos = agent["position"]
    if pos is not None:
        # Convert quaternion to rotation matrix and extract the forward (x-axis) direction.
        R = quaternion_to_matrix(agent["orientation"])
        forward = R[:, 0]  # first column
        ax2.quiver(pos[0], pos[1], pos[2],
                   forward[0], forward[1], forward[2],
                   length=arrow_scale, color='red', normalize=True)

# For each noisy agent, draw a quiver arrow.
for agent in sorted_noisy:
    pos = agent["position"]
    if pos is not None:
        # Use the noisy orientation matrix if available; otherwise, convert from the quaternion.
        R = agent.get("orientation_matrix")
        if R is None:
            R = quaternion_to_matrix(agent["orientation"])
        forward = R[:, 0]  # forward direction
        ax2.quiver(pos[0], pos[1], pos[2],
                   forward[0], forward[1], forward[2],
                   length=arrow_scale, color='cyan', normalize=True)

# Set equal axis scaling.
x_limits = ax2.get_xlim3d()
y_limits = ax2.get_ylim3d()
z_limits = ax2.get_zlim3d()
x_range = abs(x_limits[1] - x_limits[0])
y_range = abs(y_limits[1] - y_limits[0])
z_range = abs(z_limits[1] - z_limits[0])
max_range = max(x_range, y_range, z_range)
x_mid = np.mean(x_limits)
y_mid = np.mean(y_limits)
z_mid = np.mean(z_limits)
ax2.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
ax2.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
ax2.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_title("Original and Noisy Agent Trajectories with Orientations")
ax2.legend()
ax2.grid(True)
plt.show()

# =======================================================================
# Plot the optimized results
# =======================================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original agent trajectory (red) and original object positions (blue)
ax.plot(agent_x, agent_y, agent_z, marker='o', color='red', label='Original Agent Trajectory')
ax.scatter(object_centroid_x, object_centroid_y, object_centroid_z, marker='s', color='blue', s=80, label='Original Object Centroid')

# Plot noisy agent and object positions (cyan and magenta, respectively).
ax.plot(noisy_agent_x, noisy_agent_y, noisy_agent_z, marker='o', color='cyan', label='Noisy Agent (Drifted)')

# Plot optimized agent trajectory (green) and optimized landmark positions (yellow)
opt_agent_x = [pose.translation()[0] for pose in optimized_agent_poses]
opt_agent_y = [pose.translation()[1] for pose in optimized_agent_poses]
opt_agent_z = [pose.translation()[2] for pose in optimized_agent_poses]
ax.plot(opt_agent_x, opt_agent_y, opt_agent_z, marker='o', color='green', label='Optimized Agent Trajectory')

opt_landmark_x = [point[0] for point in optimized_landmark_positions]
opt_landmark_y = [point[1] for point in optimized_landmark_positions]
opt_landmark_z = [point[2] for point in optimized_landmark_positions]
ax.scatter(opt_landmark_x, opt_landmark_y, opt_landmark_z, marker='s', color='yellow', s=80, label='Optimized Landmarks')

# Add bounding boxes for original objects
# Draw bounding boxes for original objects (optional)
for obj in objects_data:
    bb = obj.get("bounding_box", None)
    if bb is not None:
        try:
            bb_min = bb["min"]
            bb_max = bb["max"]
            if bb_min.size < 3 or bb_max.size < 3:
                continue
            x_min, y_min, z_min = bb_min[:3]
            x_max, y_max, z_max = bb_max[:3]
            vertices = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ])
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[3], vertices[0], vertices[4], vertices[7]]
            ]
            poly3d = Poly3DCollection(faces, edgecolors='green', facecolors=(0, 0, 0, 0), linewidths=1)
            ax.add_collection3d(poly3d)
        except Exception as e:
            print("Failed to add bounding box:", e)

# Overlay the prior point clouds
for obj in objects_data:
    if "pcd" in obj and obj["pcd"] is not None:
        pcd_points = np.asarray(obj["pcd"].points)
        ax.scatter(pcd_points[:,0], pcd_points[:,1], pcd_points[:,2],
                   s=2, color='black', alpha=0.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Landmark SLAM Optimization Result")
ax.legend()
ax.grid(True)
plt.show()