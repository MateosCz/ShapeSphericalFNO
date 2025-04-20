import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import matplotlib.collections as mcoll

def plot_trajectory_2d(trajectory, title, trajectory_alpha=0.8, start_shape_name='start', end_shape_name='end', simplified=True):
    # trajectory: (time_steps, landmark_num, 2)
    fig, ax = plt.subplots()
    color_range = jnp.linspace(0, 1, trajectory.shape[0])
    # adjust the point size by the landmark number
    point_size = 6 + 2 * jnp.arange(trajectory.shape[1])

    # adjust the size of the plot by the landmark number
    fig.set_size_inches(10, 10)
    # color_range = jnp.flip(color_range)
    if not simplified:
        for i in range(trajectory.shape[1]):  # iterate over landmark number dimension
            x = trajectory[:,i,0]
            y = trajectory[:,i,1]
            # Reshape segments to be pairs of points
            points = jnp.stack([x, y], axis=1)  # shape: (time_steps, 2)
            segments = jnp.stack([points[:-1], points[1:]], axis=1)  # shape: (time_steps-1, 2, 2)      
        
            lc = mcoll.LineCollection(segments, cmap=plt.cm.coolwarm, alpha=trajectory_alpha)
            lc.set_array(color_range[:-1])  # one less than points due to segments
            ax.add_collection(lc)
        # plot the shape of the starting point and the ending point and connect them
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], 'o', color=plt.cm.coolwarm(color_range[0]), alpha=0.9, markersize=4)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], 'o', color=plt.cm.coolwarm(color_range[-1]), alpha=0.9, markersize=4)
        
        # add yellow cross mark to make the starting point and the ending point more visible
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], '+', color='yellow', alpha=0.7, markersize=6)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], '+', color='yellow', alpha=0.7, markersize=6)

        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.9, label=start_shape_name)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.9, label=end_shape_name)
        
        start_point_x0 = jnp.array([trajectory[0, 0, 0], trajectory[0, 0, 1]])
        end_point_x0 = jnp.array([trajectory[0, -1, 0], trajectory[0, -1, 1]])
        start_point_xT = jnp.array([trajectory[-1, 0, 0], trajectory[-1, 0, 1]])
        end_point_xT = jnp.array([trajectory[-1, -1, 0], trajectory[-1, -1, 1]])
        envelope_x0 = jnp.array([start_point_x0, end_point_x0])
        envelope_xT = jnp.array([start_point_xT, end_point_xT])
        # add label to the starting point and the ending point
        # ax.text(start_point_x0[0], start_point_x0[1], start_shape_name, color=plt.cm.coolwarm(color_range[0]), alpha=0.9)
        # ax.text(end_point_x0[0], end_point_x0[1], end_shape_name, color=plt.cm.coolwarm(color_range[-1]), alpha=0.9)
        ax.legend()
        ax.plot(envelope_x0[:, 0], envelope_x0[:, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.7)
        ax.plot(envelope_xT[:, 0], envelope_xT[:, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7)
        
        # add color bar
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), ax=ax, orientation='vertical')
        cbar.set_label('Time')
    else:
        for i in range(trajectory.shape[1]):  # iterate over landmark number dimension
            ax.plot(trajectory[:,i,0], trajectory[:,i,1], '-', color='orange', alpha=trajectory_alpha)
        # plot the shape of the starting point and the ending point and connect them
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], 'o', color=plt.cm.coolwarm(color_range[0]), alpha=0.7, markersize=6)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], 'o', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7, markersize=6)
        # add yellow cross mark to make the starting point and the ending point more visible
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], '+', color='yellow', alpha=0.7, markersize=6)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], '+', color='yellow', alpha=0.7, markersize=6)

        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.7, label=start_shape_name)# connect the landmarks at the start time
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7, label=end_shape_name)# connect the landmarks at the end time
        # envelope the start and end points
        start_point_x0 = jnp.array([trajectory[0, 0, 0], trajectory[0, 0, 1]])
        end_point_x0 = jnp.array([trajectory[0, -1, 0], trajectory[0, -1, 1]])
        start_point_xT = jnp.array([trajectory[-1, 0, 0], trajectory[-1, 0, 1]])
        end_point_xT = jnp.array([trajectory[-1, -1, 0], trajectory[-1, -1, 1]])
        envelope_x0 = jnp.array([start_point_x0, end_point_x0])
        envelope_xT = jnp.array([start_point_xT, end_point_xT])
        # add label to the starting point and the ending point and place the label at the corner of the plot
        ax.legend()
        ax.plot(envelope_x0[:, 0], envelope_x0[:, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.7)
        ax.plot(envelope_xT[:, 0], envelope_xT[:, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7)


    
    # keep the scale of x and y the same
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    plt.show()

#plot the score for a group of landmarks, use quiver to plot the vector field
def plot_score_field(score, t, x0, 
                     xmin=-2.0, xmax=2.0, num_x=20,
                     ymin=-2.0, ymax=2.0, num_y=20,
                     landmark_to_plot=0):
    """
    Plot a score field for a given score function where the score function is defined as:
        score(x, t, x0)
    with:
        x: (landmark_num, 2) array of landmarks coordinates
        t: scalar
        x0: (landmark_num, 2) fixed reference landmarks

    Parameters
    ----------
    score : function
        A function score(x, t, x0) -> (landmark_num, 2)
    t : scalar
        A scalar parameter for the score function.
    x0 : jnp.ndarray, shape (landmark_num, 2)
        Reference landmark positions.
    xmin, xmax, ymin, ymax : float
        Defines the 2D region over which we plot the score field.
    num_x, num_y : int
        Number of points along x and y directions to form the grid.
    landmark_to_plot : int
        Index of the landmark in the output to visualize.

    Returns
    -------
    None
        Displays a quiver plot of the selected landmark's score field.
    """

    # Generate a grid of points
    xs = jnp.linspace(xmin, xmax, num_x)
    ys = jnp.linspace(ymin, ymax, num_y)
    X, Y = jnp.meshgrid(xs, ys)

    # Flatten the grid into (num_points, 2)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (num_points, 2)

    # Function to create an input x array for each point
    # We vary the position of the chosen landmark along the grid,
    # while keeping other landmarks the same as x0.
    def make_x_for_point(x_val, y_val):
        modified_x = x0.at[landmark_to_plot, 0].set(x_val)
        modified_x = modified_x.at[landmark_to_plot, 1].set(y_val)
        return modified_x

    create_x_array_for_point = jax.vmap(make_x_for_point, in_axes=(0,0))
    X_array = create_x_array_for_point(points[:,0], points[:,1])
    # X_array: (num_points, landmark_num, 2)

    # Vectorized score evaluation
    def batch_score(x):
        return score(x, t, x0)  # shape: (landmark_num, 2)

    S = jax.vmap(batch_score, in_axes=(0,))(X_array) 
    # S: (num_points, landmark_num, 2)

    # Extract the vector field for the chosen landmark
    U = S[:, landmark_to_plot, 0]
    V = S[:, landmark_to_plot, 1]

    # Reshape to (num_y, num_x)
    U = U.reshape(X.shape)
    V = V.reshape(Y.shape)

    # Plot using quiver
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=20, color='red', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Score Field at t={t}')
    plt.axis('equal')
    plt.show()


#plot the matrix and show the heatmap with the color bar
def plot_matrix(matrix, title):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='coolwarm')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.show()


    
    
def plot_trajectory_3d(trajectory, title, trajectory_alpha=0.8, start_shape_name='start', end_shape_name='end', simplified=True, perspective='auto'):
    # trajectory: (time_steps, landmark_num, 3) - 3D trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    color_range = jnp.linspace(0, 1, trajectory.shape[0])
    if perspective == 'auto':
        pass
    elif perspective == 'x':
        ax.view_init(elev=0, azim=0) 
    elif perspective == 'y':
        ax.view_init(elev=0, azim=90)
    elif perspective == 'z':
        ax.view_init(elev=90, azim=0)
        
    if not simplified:
        for i in range(trajectory.shape[1]):  # iterate over landmark number dimension
            x = trajectory[:,i,0]
            y = trajectory[:,i,1]
            z = trajectory[:,i,2]
            
            # Plot the trajectory line for each landmark
            points = jnp.column_stack([x, y, z])
            ax.plot(x, y, z, '-', color='orange', alpha=trajectory_alpha)
            
            # Color the line according to time
            for t in range(len(x)-1):
                ax.plot(x[t:t+2], y[t:t+2], z[t:t+2], color=plt.cm.coolwarm(color_range[t]), alpha=trajectory_alpha)
        
        # Plot start and end points
        ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], trajectory[0, :, 2], 
                  color=plt.cm.coolwarm(color_range[0]), alpha=0.9, s=30, marker='o')
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], trajectory[-1, :, 2], 
                  color=plt.cm.coolwarm(color_range[-1]), alpha=0.9, s=30, marker='o')
        
        # Add yellow markers to make start/end points more visible
        ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], trajectory[0, :, 2], 
                  color='yellow', alpha=0.7, s=40, marker='+')
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], trajectory[-1, :, 2], 
                  color='yellow', alpha=0.7, s=40, marker='+')
        
        # # Connect landmarks at start and end times
        # for i in range(trajectory.shape[1]-1):
        #     # Connect start shape
        #     ax.plot([trajectory[0, i, 0], trajectory[0, i+1, 0]], 
        #             [trajectory[0, i, 1], trajectory[0, i+1, 1]], 
        #             [trajectory[0, i, 2], trajectory[0, i+1, 2]], 
        #             '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.9)
            
        #     # Connect end shape
        #     ax.plot([trajectory[-1, i, 0], trajectory[-1, i+1, 0]], 
        #             [trajectory[-1, i, 1], trajectory[-1, i+1, 1]], 
        #             [trajectory[-1, i, 2], trajectory[-1, i+1, 2]], 
        #             '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.9)
        
        # Add a colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), ax=ax, orientation='vertical')
        cbar.set_label('Time')
        
        # Add legend
        ax.scatter([], [], [], color=plt.cm.coolwarm(color_range[0]), marker='o', 
                  s=30, label=start_shape_name)
        ax.scatter([], [], [], color=plt.cm.coolwarm(color_range[-1]), marker='o', 
                  s=30, label=end_shape_name)
        ax.legend()
        
    else:
        # Simplified version
        for i in range(trajectory.shape[1]):  # iterate over landmark number dimension
            ax.plot(trajectory[:,i,0], trajectory[:,i,1], trajectory[:,i,2], 
                   '-', color='orange', alpha=trajectory_alpha)
        
        # Plot start and end points
        ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], trajectory[0, :, 2], 
                  color=plt.cm.coolwarm(color_range[0]), alpha=0.7, s=40, marker='o')
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], trajectory[-1, :, 2], 
                  color=plt.cm.coolwarm(color_range[-1]), alpha=0.7, s=40, marker='o')
        
        # Add yellow markers
        ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], trajectory[0, :, 2], 
                  color='yellow', alpha=0.7, s=40, marker='+')
        ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], trajectory[-1, :, 2], 
                  color='yellow', alpha=0.7, s=40, marker='+')
        
        # # Connect landmarks at start and end times
        # for i in range(trajectory.shape[1]-1):
        #     # Connect start shape
        #     ax.plot([trajectory[0, i, 0], trajectory[0, i+1, 0]], 
        #             [trajectory[0, i, 1], trajectory[0, i+1, 1]], 
        #             [trajectory[0, i, 2], trajectory[0, i+1, 2]], 
        #             '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.7)
            
        #     # Connect end shape
        #     ax.plot([trajectory[-1, i, 0], trajectory[-1, i+1, 0]], 
        #             [trajectory[-1, i, 1], trajectory[-1, i+1, 1]], 
        #             [trajectory[-1, i, 2], trajectory[-1, i+1, 2]], 
        #             '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7)
        
        # Add legend
        ax.scatter([], [], [], color=plt.cm.coolwarm(color_range[0]), marker='o', 
                  s=30, label=start_shape_name)
        ax.scatter([], [], [], color=plt.cm.coolwarm(color_range[-1]), marker='o', 
                  s=30, label=end_shape_name)
        ax.legend()
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    # This is a bit tricky in 3D, so we'll set limits based on data range
    x_range = jnp.ptp(trajectory[:,:,0])
    y_range = jnp.ptp(trajectory[:,:,1])
    z_range = jnp.ptp(trajectory[:,:,2])
    max_range = jnp.max(jnp.array([x_range, y_range, z_range]))
    
    x_mid = jnp.mean(trajectory[:,:,0])
    y_mid = jnp.mean(trajectory[:,:,1])
    z_mid = jnp.mean(trajectory[:,:,2])
    
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    
    plt.show()

# plot 3d trajectory in polyscope
def plot_trajectory_3d_polyscope(trajectory, current_frame, title, trajectory_alpha=0.5,simplified=True):
    # trajectory: (time_steps, landmark_num, 3) - 3D trajectory
    import polyscope as ps  
    import polyscope.imgui as psimgui
    import numpy as np
    color_range = jnp.linspace(0, 1, trajectory.shape[0])
    nodes = []
    edges = []
    colors = []
    node_scale = []
    edge_scale = []
    for t in range(trajectory.shape[0]):
        if t == current_frame:
            break
        color = plt.cm.coolwarm(color_range[t])
        for i in range(trajectory.shape[1]):
            nodes.append(trajectory[t, i])
            node_scale.append(0.001)
            if t > 0:
                edges.append(np.array([t * trajectory.shape[1] + i, (t-1) * trajectory.shape[1] + i]))
                edge_scale.append(0.02)
                colors.append(color[:3])
    nodes = np.array(nodes)
    edges = np.array(edges)
    colors = np.array(colors)
    node_scale = np.array(node_scale)
    edge_scale = np.array(edge_scale)
    print(nodes.shape, edges.shape, colors.shape)
    if len(nodes) > 0 and len(edges) > 0 and len(colors) > 0:
        ps_curve = ps.register_curve_network("trajectory", nodes, edges)
        ps_curve.add_color_quantity("color", colors, defined_on="edges", enabled=True)
        ps_curve.set_radius(0.0012)
        ps_curve.set_transparency(0.2)
        # ps_curve.add_scalar_quantity("node_scale", node_scale, enabled=True)
        # ps_curve.add_scalar_quantity("edge_scale", edge_scale, defined_on="edges")


import polyscope as ps
import polyscope.imgui as psim
import numpy as np

def visualize_score_field_over_time(score_lst, positions, dt=0.01, scale=0.3, radius=1.0):
    """
    用 Polyscope 可视化单位球上的 score 向量场，随时间变化。

    Args:
        score_lst: (T, n_lat, n_lon, 3)，每一帧的向量场
        positions: (T, n_lat, n_lon, 3)，对应每一帧的球面点位置
        dt: 时间步长
        scale: 向量缩放比例
        radius: 背景球半径
    """
    ps.init()
    ps.remove_all_structures()

    total_time = score_lst.shape[0] * dt
    time = 0.0
    frame_idx = 0

    def update_frame(idx):
        ps.remove_all_structures()

        pts = positions[idx].reshape(-1, 3)
        vecs = score_lst[idx].reshape(-1, 3)

        pc = ps.register_point_cloud("points", pts)
        pc.set_radius(0.005)
        pc.add_vector_quantity("score", vecs * scale, enabled=True)

        # # 添加背景球
        # u = np.linspace(0, 2 * np.pi, 40)
        # v = np.linspace(0, np.pi, 20)
        # x = radius * np.outer(np.cos(u), np.sin(v))
        # y = radius * np.outer(np.sin(u), np.sin(v))
        # z = radius * np.outer(np.ones_like(u), np.cos(v))
        # sphere = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        # ps.register_point_cloud("unit_sphere", sphere)
        # ps.get_point_cloud("unit_sphere").set_radius(0.001)
        # ps.get_point_cloud("unit_sphere").set_color((0.8, 0.8, 0.8))
        # ps.get_point_cloud("unit_sphere").set_transparency(0.95)

    update_frame(frame_idx)

    def callback():
        nonlocal time, frame_idx
        changed, time = psim.SliderFloat("Time", time, v_min=0.0, v_max=total_time)
        if changed:
            frame_idx = int(time / dt)
            frame_idx = min(frame_idx, score_lst.shape[0] - 1)
            time = frame_idx * dt
            update_frame(frame_idx)

    ps.set_user_callback(callback)
    ps.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_score_field_with_regions(score_lst, positions, dt=0.01, scale=0.3, radius=1.0):
    import polyscope as ps
    import polyscope.imgui as psim
    import numpy as np
    import matplotlib.cm as cm

    ps.init()
    ps.remove_all_structures()
    total_time = score_lst.shape[0] * dt
    time = 0.0
    frame_idx = 0

    def update_frame(idx):
        ps.remove_all_structures()
        pts = positions[idx].reshape(-1, 3)
        vecs = score_lst[idx].reshape(-1, 3)
        norms = np.linalg.norm(vecs, axis=-1)

        # Colormap for vectors
        norm_min, norm_max = norms.min(), norms.max()
        norm_scaled = (norms - norm_min) / (norm_max - norm_min + 1e-8)
        colors = cm.coolwarm(norm_scaled)[:, :3]

        # Region coloring
        z = pts[:, 2]
        theta = np.arccos(np.clip(z / np.linalg.norm(pts, axis=-1), -1.0, 1.0))
        region_colors = np.full((pts.shape[0], 3), 0.7)
        region_colors[(theta < 0.2) | (theta > np.pi - 0.2)] = [1.0, 0.3, 0.3]
        region_colors[(np.abs(theta - np.pi/2) < 0.1)] = [0.3, 1.0, 0.3]

        # Point cloud with vector and color quantities
        pc = ps.register_point_cloud("points", pts)
        pc.set_radius(0.005)
        pc.add_vector_quantity("score", vecs * scale, enabled=True)
        pc.add_color_quantity("region", region_colors, enabled=True)

        # # Add background sphere
        # u = np.linspace(0, 2 * np.pi, 40)
        # v = np.linspace(0, np.pi, 20)
        # x = radius * np.outer(np.cos(u), np.sin(v))
        # y = radius * np.outer(np.sin(u), np.sin(v))
        # z = radius * np.outer(np.ones_like(u), np.cos(v))
        # sphere = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        # bg = ps.register_point_cloud("unit_sphere", sphere)
        # bg.set_radius(0.001)
        # bg.set_color((0.8, 0.8, 0.8))
        # bg.set_transparency(0.95)

        # === 坐标轴 ===
        axis_length = 1.5
        x_axis = np.array([[0, 0, 0], [axis_length, 0, 0]])
        y_axis = np.array([[0, 0, 0], [0, axis_length, 0]])
        z_axis = np.array([[0, 0, 0], [0, 0, axis_length]])
        ps.register_curve_network("x-axis", x_axis, np.array([[0, 1]])).set_color((1, 0, 0))
        ps.register_curve_network("y-axis", y_axis, np.array([[0, 1]])).set_color((0, 1, 0))
        ps.register_curve_network("z-axis", z_axis, np.array([[0, 1]])).set_color((0, 0, 1))

    update_frame(frame_idx)

    def callback():
        nonlocal time, frame_idx
        changed, time = psim.SliderFloat("Time", time, v_min=0.0, v_max=total_time)
        if changed:
            frame_idx = int(time / dt)
            frame_idx = min(frame_idx, score_lst.shape[0] - 1)
            time = frame_idx * dt
            update_frame(frame_idx)

    ps.set_user_callback(callback)
    ps.show()
