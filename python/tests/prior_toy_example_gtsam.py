import spark_dsg as dsg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gtsam
from gtsam import Pose3, Rot3, Point3, symbol, NonlinearFactorGraph, Values, noiseModel

np.random.seed(14)

# =============================================================================
# Load the Scene Graph and Remove Nodes
# =============================================================================
path_to_dsg = "example_dsg_for_jared.json"
G = dsg.DynamicSceneGraph.load(path_to_dsg)

########## Remove Some Object Nodes
object_layer = G.get_layer(dsg.DsgLayers.OBJECTS)
object_nodes = list(object_layer.nodes)
nodes_to_remove = object_nodes[:3] + object_nodes[-3:]
for node in nodes_to_remove:
    G.remove_node(node.id.value)

############# Remove Some Agent Nodes
agent_layer = G.get_dynamic_layer(dsg.DsgLayers.AGENTS, "a")
agent_nodes = list(agent_layer.nodes)
nodes_to_remove = agent_nodes[:300] + agent_nodes[-100:]
for node in nodes_to_remove:
    G.remove_node(node.id.value)

# =============================================================================
# Step 2: Extract Original (Prior) Data
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
    traj_positions = obj.attributes.trajectory_positions if hasattr(obj.attributes, "trajectory_positions") else None
    traj_timestamps = obj.attributes.trajectory_timestamps if hasattr(obj.attributes, "trajectory_timestamps") else None
    if pos is not None:
        pos = np.array(pos).flatten()
    objects_data.append({
        "id": obj.id,
        "position": pos,
        "orientation": R_object,
        "bounding_box": bounding_box,
        "trajectory_positions": traj_positions,
        "trajectory_timestamps": traj_timestamps
    })
# =============================================================================
# Step 3: Create Measurement Edges between Original (Prior) Agents and Objects
# =============================================================================
radius_threshold = 3
measurement_edges = []
for obj in objects_data:
    obj_pos = obj["position"]
    if obj_pos is None:
        continue
    for agent in agent_trajectories:
        agent_pos = agent["position"]
        if agent_pos is None:
            continue
        distance = np.linalg.norm(obj_pos - agent_pos)
        if distance <= radius_threshold:
            edge = {
                "object_id": obj["id"],
                "agent_id": agent["id"],
                "distance": distance,
                "object_position": obj_pos,
                "agent_position": agent_pos
            }
            measurement_edges.append(edge)

# For visualization
agent_x = [a["position"][0] for a in agent_trajectories if a["position"] is not None]
agent_y = [a["position"][1] for a in agent_trajectories if a["position"] is not None]
agent_z = [a["position"][2] for a in agent_trajectories if a["position"] is not None]

object_x = [o["position"][0] for o in objects_data if o["position"] is not None]
object_y = [o["position"][1] for o in objects_data if o["position"] is not None]
object_z = [o["position"][2] for o in objects_data if o["position"] is not None]

# =============================================================================
# Step 4: Define Helper Functions to Add Noise and Simulate Cumulative Drift
# =============================================================================
def perturb_quaternion(q, noise_std):
    w, x, y, z = q.w, q.x, q.y, q.z
    angle = 2 * np.arccos(w)
    sin_half_angle = np.sqrt(max(0, 1 - w*w))
    if sin_half_angle < 1e-6:
        axis = np.array([1, 0, 0])
    else:
        axis = np.array([x, y, z]) / sin_half_angle
    delta_angle = np.random.normal(0, noise_std)
    new_angle = angle + delta_angle
    new_w = np.cos(new_angle/2)
    new_xyz = axis * np.sin(new_angle/2)
    return type(q)(new_w, new_xyz[0], new_xyz[1], new_xyz[2])

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

def add_drift_and_noise_to_objects(objects, global_drift, measurement_noise_std=0.05):
    noisy_objects = []
    for obj in objects:
        noisy_obj = obj.copy()
        if noisy_obj["position"] is not None:
            noisy_obj["position"] = noisy_obj["position"] + global_drift + np.random.normal(0, measurement_noise_std, 3)
        noisy_objects.append(noisy_obj)
    return noisy_objects

# =============================================================================
# Step 5: Create Noisy Copies and Measurement Edges for Noisy Data
# =============================================================================
noisy_agent_trajectories = add_cumulative_drift_to_agents(agent_trajectories, drift_std=0.05, alpha=0.9)


# =============================================================================
# Step 6: Build a GTSAM Factor Graph
# =============================================================================
graph = NonlinearFactorGraph()
initial_estimates = Values()

sorted_agents = sorted(agent_trajectories, key=lambda x: x["timestamp"])
sorted_noisy_agents = sorted(noisy_agent_trajectories, key=lambda x: x["timestamp"])
n_agents = len(sorted_noisy_agents)
n_landmarks = len(objects_data)

# Create initial estimates for agent poses.
agent_poses = []
for i, agent in enumerate(sorted_noisy_agents):
    q = agent["orientation"]
    R = Rot3.Quaternion(q.w, q.x, q.y, q.z)
    t = Point3(agent["position"][0], agent["position"][1], agent["position"][2])
    pose = Pose3(R, t)
    agent_poses.append(pose)
    key = symbol('x', i)
    initial_estimates.insert(key, pose)

# Sanity check using the original agent poses.
# agent_poses = []
# for i, agent in enumerate(sorted_agents):
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
    agent_index = next((i for i, a in enumerate(sorted_agents) if a["id"] == edge["agent_id"]), None)
    # Find the corresponding landmark in the noisy objects.
    landmark_index = next((j for j, o in enumerate(objects_data) if o["id"] == edge["object_id"]), None)
    if agent_index is None or landmark_index is None:
        continue
    key_agent = symbol('x', agent_index)
    key_landmark = symbol('l', landmark_index)
    # Construct Pose3 for the noisy agent.
    agent = sorted_agents[agent_index]
    q_a = agent["orientation"]
    agent_pose = Pose3(Rot3.Quaternion(q_a.w, q_a.x, q_a.y, q_a.z),
                             Point3(agent["position"][0],
                                    agent["position"][1],
                                    agent["position"][2]))
    # Construct Pose3 for the landmark.
    object = next((o for o in objects_data if o["id"] == edge["object_id"]), None)
    if object is None:
        continue
    # Assume the landmark has identity rotation.
    landmark_pose_noisy = Pose3(Rot3(), Point3(object["position"][0],
                                               object["position"][1],
                                               object["position"][2]))
    # Compute the relative measurement: T_agent^{-1} * T_landmark.
    relative_measurement = agent_pose.between(landmark_pose_noisy) # TODO: Add noise to this measurement.
    graph.add(gtsam.BetweenFactorPose3(key_agent, key_landmark, relative_measurement, relative_noise))

# Add prior factors for landmarks using the original objects_data.
prior_noise = noiseModel.Isotropic.Sigma(6, 0.1)  # 6-dim noise.
for j, obj in enumerate(objects_data):
    key = symbol('l', j)
    prior = Pose3(Rot3(), Point3(obj["position"][0], obj["position"][1], obj["position"][2]))
    graph.add(gtsam.PriorFactorPose3(key, prior, prior_noise))

# =============================================================================
# Step 7: Optimize the Factor Graph
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
# (Optional) Visualization of Priors and Optimized Results
# =============================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original agent trajectory (red) and object positions (blue)
ax.plot(agent_x, agent_y, agent_z, marker='o', color='red', label='Original Agent Trajectory')
ax.scatter(object_x, object_y, object_z, marker='s', color='blue', s=80, label='Original Object Position')

# plot original measurement edges as purple dashed lines.
for edge in measurement_edges:
    p_obj = edge["object_position"]
    p_agent = edge["agent_position"]
    ax.plot([p_obj[0], p_agent[0]],
            [p_obj[1], p_agent[1]],
            [p_obj[2], p_agent[2]],
            color='purple', linestyle='--', linewidth=1)

# For noisy data, extract positions for plotting.
noisy_agent_x = [a["position"][0] for a in noisy_agent_trajectories if a["position"] is not None]
noisy_agent_y = [a["position"][1] for a in noisy_agent_trajectories if a["position"] is not None]
noisy_agent_z = [a["position"][2] for a in noisy_agent_trajectories if a["position"] is not None]

# Plot noisy agent and object positions (cyan and magenta, respectively).
ax.plot(noisy_agent_x, noisy_agent_y, noisy_agent_z, marker='o', color='cyan', label='Noisy Agent (Drifted)')
# ax.scatter(noisy_object_x, noisy_object_y, noisy_object_z, marker='s', color='magenta', s=80, label='Noisy Object (Drift+Noise)')

# Plot noisy measurement edges as orange dotted lines.
# for edge in noisy_measurement_edges:
#     p_obj = edge["object_position"]
#     p_agent = edge["agent_position"]
#     ax.plot([p_obj[0], p_agent[0]],
#             [p_obj[1], p_agent[1]],
#             [p_obj[2], p_agent[2]],
#             color='orange', linestyle=':', linewidth=1)

# Draw bounding boxes for original objects (optional)
for obj in objects_data:
    bb = obj.get("bounding_box", None)
    if bb is not None:
        try:
            if isinstance(bb, dict):
                bb_min = np.array(bb["pos"]) - np.array(bb["dim"]) / 2
                bb_max = np.array(bb["pos"]) + np.array(bb["dim"]) / 2
            else:
                bb_min = np.array(bb.min).flatten()
                bb_max = np.array(bb.max).flatten()
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
# Plot the optimized results
# =======================================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original agent trajectory (red) and original object positions (blue)
ax.plot(agent_x, agent_y, agent_z, marker='o', color='red', label='Original Agent Trajectory')
ax.scatter(object_x, object_y, object_z, marker='s', color='blue', s=80, label='Original Object Position')

noisy_agent_x = [a["position"][0] for a in noisy_agent_trajectories if a["position"] is not None]
noisy_agent_y = [a["position"][1] for a in noisy_agent_trajectories if a["position"] is not None]
noisy_agent_z = [a["position"][2] for a in noisy_agent_trajectories if a["position"] is not None]

# noisy_object_x = [o["position"][0] for o in noisy_objects_data if o["position"] is not None]
# noisy_object_y = [o["position"][1] for o in noisy_objects_data if o["position"] is not None]
# noisy_object_z = [o["position"][2] for o in noisy_objects_data if o["position"] is not None]

# Plot noisy agent and object positions (cyan and magenta, respectively).
ax.plot(noisy_agent_x, noisy_agent_y, noisy_agent_z, marker='o', color='cyan', label='Noisy Agent (Drifted)')
# ax.scatter(noisy_object_x, noisy_object_y, noisy_object_z, marker='s', color='magenta', s=80, label='Noisy Object (Drift+Noise)')

# Plot optimized agent trajectory (green) and optimized landmark positions (yellow)
opt_agent_x = [pose.translation()[0] for pose in optimized_agent_poses]
opt_agent_y = [pose.translation()[1] for pose in optimized_agent_poses]
opt_agent_z = [pose.translation()[2] for pose in optimized_agent_poses]
ax.plot(opt_agent_x, opt_agent_y, opt_agent_z, marker='o', color='green', label='Optimized Agent Trajectory')

opt_landmark_x = [point[0] for point in optimized_landmark_positions]
opt_landmark_y = [point[1] for point in optimized_landmark_positions]
opt_landmark_z = [point[2] for point in optimized_landmark_positions]
ax.scatter(opt_landmark_x, opt_landmark_y, opt_landmark_z, marker='s', color='yellow', s=80, label='Optimized Landmarks')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Landmark SLAM Optimization Result")
ax.legend()
ax.grid(True)
plt.show()

# =============================================================================
# (Optional) Visualize the Factor Graph by exporting it in DOT format.
# =============================================================================
# dot_str = graph.dot()
# with open("factor_graph.dot", "w") as f:
#     f.write(dot_str)
# print("Factor graph DOT file written to 'factor_graph.dot'.")
