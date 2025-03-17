# viz_open3d.py
import numpy as np
import spark_dsg as dsg


# path_to_dsg = "spot_building45_gadget.json"
path_to_dsg = "out_dsg.json"
# path_to_dsg = "t3_w0_ths2_fused.json"
# path_to_dsg = "example_dsg_for_jared.json"
# path_to_dsg = "apartment_dsg.json"

G = dsg.DynamicSceneGraph.load(str(path_to_dsg))

# %%
print("Layers:")
for layer in G.layers:
    print(f"  - {layer.id}")
print("")

# %%
room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
room_node_strs = [f"{x.id}" for x in room_layer.nodes]
print(f"Rooms: {room_node_strs}")

room_edge_strs = [f"  - {x}" for x in room_layer.edges]
print("Room Edges:")
print("\n".join(room_edge_strs))
print("")


# ## Interlayer Edge Access

# %%
layer_edge_counts = {}
for edge in G.interlayer_edges:
    source_layer = G.get_node(edge.source).layer
    target_layer = G.get_node(edge.target).layer
    if source_layer not in layer_edge_counts:
        layer_edge_counts[source_layer] = {}
    if target_layer not in layer_edge_counts[source_layer]:
        layer_edge_counts[source_layer][target_layer] = 0

    layer_edge_counts[source_layer][target_layer] += 1

print("Interlayer Edges:")
for source_layer in layer_edge_counts:
    print(f"  - {source_layer} -> {layer_edge_counts[source_layer]}")
print("")

# %% [markdown]
# ## Entire Graph Access

# %%
node_type_counts = {}
for node in G.nodes:
    if node.id.category not in node_type_counts:
        node_type_counts[node.id.category] = 0

    node_type_counts[node.id.category] += 1

print("Node Types:")
for category, count in node_type_counts.items():
    print(f"  - {category}: {count}")


edge_counts = {}
for edge in G.edges:
    source_layer = G.get_node(edge.source).layer
    target_layer = G.get_node(edge.target).layer
    if source_layer not in edge_counts:
        edge_counts[source_layer] = {}
    if target_layer not in edge_counts[source_layer]:
        edge_counts[source_layer][target_layer] = 0

    edge_counts[source_layer][target_layer] += 1

print("All Edges:")
for source_layer in edge_counts:
    print(f"  - {source_layer} -> {edge_counts[source_layer]}")
print("")

# %% [markdown]
# ## Bounding boxes from nodes in lower layers

# %%
# dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
# print("Room bounding boxes:")
# for node in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
#     print(f"  - {node.id}: {node.attributes.bounding_box}")


dsg.render_to_open3d(G)

# fig = dsg.plot_scene_graph(G)
# if fig is not None:
#     fig.show(renderer="browser")

    