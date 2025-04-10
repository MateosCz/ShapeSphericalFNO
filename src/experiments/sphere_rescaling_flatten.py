import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import polyscope.imgui as psim
import scipy as sp
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as nn
import jax.lax as lax
import os
from src.dataGenerator.spherical_data_generator import *
from src.stochastics.sde import *
from src.stochastics.sde_solver import *
import random
import polyscope as ps
import src.training.trainer as Trainer
import matplotlib.pyplot as plt
from src.utils.plotting import plot_trajectory_3d_polyscope, plot_trajectory_3d
from src.models.neural_operator import CTShapeSFNO
from flax.training import checkpoints
def get_random_int():
    return random.randint(0, 1000000)
cwd = os.getcwd()


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))



if __name__ == "__main__":
    # jax.clear_caches()
    train_steps = 300
    retrain = False
    retrain_steps = 500
    draw_unconditional = False
    in_grid_L = 10
    sphere_data_generator_XT = S2ManifoldDataGenerator(seed=get_random_int(), radius=0.5, flatten=True)

    xT = sphere_data_generator_XT.generate_data(in_grid_L, 1)
    print(xT.shape)
    sphere_data_generator_X0 = S2ManifoldDataGenerator(radius=0.7, seed=get_random_int(), flatten=True)
    x0 = sphere_data_generator_X0.generate_data(in_grid_L, 5)
    print(x0.shape)
    sde_3d = Brownian_Motion_SDE_Flatten(dim=3, sigma=0.1, x0=x0[0])
    # sde_3d = Kunita_Flow_SDE_3D_Eulerian(k_alpha=1.6, k_sigma=0.4, grid_num=10, grid_range=[-1,1], x0=x0[0])
    sde_solver = EulerMaruyama.from_sde(sde_3d, 0.02, 1.0, 3, None,debug_mode=False)
    xs,_ = sde_solver.solve(x0[0], rng_key=jrandom.PRNGKey(get_random_int()))

    
    if not draw_unconditional:
        model = CTShapeSFNO(x_feature_dim=3, l_list=(10, 6,3), lift_dim=8, latent_feature_dims=(4, 6,8), sampling="dh", activation="gelu")
        trainer = Trainer.NeuralOpTrainer(seed=get_random_int(), landmark_num=in_grid_L)

        checkpoint_path = project_root() + '/checkpoints/sphere_model_neuralOp_1D_brownian'
        retrain_checkpoint_path = project_root() + '/checkpoints/sphere_model_retrain_neuralOp_1D_brownian'
    
        if not os.path.exists(checkpoint_path):
            train_state = trainer.train_state_init(model, lr=1e-4, model_kwargs={'x': jax.random.normal(jrandom.PRNGKey(get_random_int()), x0[0].shape), 't': jnp.array([0]),'object_fn': 'infinite', 'x_L': in_grid_L})
            train_state, train_loss = trainer.train(train_state, sde_3d, sde_solver, sphere_data_generator_X0, train_steps, 16, x_L=in_grid_L)
            plt.plot(train_loss)
            plt.show()
            # save the model
            config = {"dimension": x0[0].shape}
            ckpt = {"model": train_state, "config": config}
            checkpoints.save_checkpoint(checkpoint_path, ckpt, step=train_steps, overwrite=True, keep=1)
        else:
            restored_checkpoint = checkpoints.restore_checkpoint(checkpoint_path, target=None)
            params = restored_checkpoint["model"]["params"]
            train_state = trainer.train_state_init(model, lr=1e-4, model_kwargs={'x': jax.random.normal(jrandom.PRNGKey(get_random_int()), x0[0].shape), 't': jnp.array([0]),'object_fn': 'infinite', 'x_L': in_grid_L}, retrain=True, ckpt_params=params)
            if retrain:
                train_state, train_loss = trainer.train(train_state, sde_3d, sde_solver, sphere_data_generator_X0, retrain_steps, 16, x_L=in_grid_L)
                plt.plot(train_loss)
                plt.show()
                # save the model
                config = {"dimension": x0[0].shape}
                ckpt = {"model": train_state, "config": config}
                checkpoints.save_checkpoint(retrain_checkpoint_path, ckpt, step=retrain_steps, overwrite=True, keep=1)
        test_in_grid_L = 12
        score_fn = lambda x, t, x0: train_state.apply_fn(train_state.params, x, t, x_L=test_in_grid_L)
        x0 = sphere_data_generator_X0.generate_data(test_in_grid_L, 5)
        xT = sphere_data_generator_XT.generate_data(test_in_grid_L, 1)   
        sde_3d = Brownian_Motion_SDE_Flatten(dim=3, sigma=0.1, x0=x0[0])
        reverse_sde = Time_Reversed_SDE_infinite(sde_3d, score_fn, 1.0,0.02)
        reverse_solver = EulerMaruyama.from_sde(reverse_sde, 0.02, 1.0, 3, condition_x=x0[0],debug_mode=False)
        condition_xs,_ = reverse_solver.solve(xT[0], rng_key=jrandom.PRNGKey(get_random_int()))
        # condition_xs = xs
        condition_xs = np.array(condition_xs)
        trajectory_xs = condition_xs
    else:
        xs = np.array(xs)
        trajectory_xs = xs
    x0 = np.array(x0)
    xT = np.array(xT)
    

    # plot_trajectory_3d(condition_xs, "reverse_trajectory_finite" + "k_alpha=1.6" + "k_sigma=0.4" + "grid_num=10" + "grid_range=[-1,1]", simplified=False, perspective='x')
    # plot_trajectory_3d(condition_xs, "reverse_trajectory_finite" + "k_alpha=1.6" + "k_sigma=0.4" + "grid_num=10" + "grid_range=[-1,1]", simplified=False, perspective='z')
    # plot_trajectory_3d(condition_xs, "reverse_trajectory_finite" + "k_alpha=1.6" + "k_sigma=0.4" + "grid_num=10" + "grid_range=[-1,1]", simplified=False, perspective='y')

     

    # Create a new figure

    ps.init()
    # global frame_idx
    time = 0.0
    total_time = 1.0
    dt = 0.02
    frame_idx = 0

    ps.set_ground_plane_mode("shadow_only") 
    ps.set_ground_plane_height_mode("manual")
    ps.set_ground_plane_height(-0.2)
    ps.set_view_projection_mode("orthographic")
    ps.look_at((2., 2., 2.), (0., 0., 0.))
    def active_animation():
        for x in trajectory_xs[0]:
            ps_cloud = ps.register_point_cloud("my points", x)

            # ps_mesh.add_scalar_quantity("scalar", xs[:, 0], enabled=True)



    def imgui_callback():
        global time
        global frame_idx
        
        frame_idx = int(time/dt)
        ps_cloud = ps.register_point_cloud("my points", trajectory_xs[frame_idx])
        axis_length = 2.0
        x_axis = np.array([[0,0,0], [axis_length,0,0]])
        y_axis = np.array([[0,0,0], [0,axis_length,0]])
        z_axis = np.array([[0,0,0], [0,0,axis_length]])

        ps.register_curve_network("x-axis", np.array([[0,0,0], [axis_length,0,0]]), np.array([[0,1]]))
        ps.register_curve_network("y-axis", np.array([[0,0,0], [0,axis_length,0]]), np.array([[0,1]]))
        ps.register_curve_network("z-axis", np.array([[0,0,0], [0,0,axis_length]]), np.array([[0,1]]))

        # Set axis colors
        ps.get_curve_network("x-axis").set_color((1,0,0))  # Red for X
        ps.get_curve_network("y-axis").set_color((0,1,0))  # Green for Y
        ps.get_curve_network("z-axis").set_color((0,0,1))  # Blue for Z

        



        changed, time = psim.SliderFloat("Time", time, v_min=0,v_max=total_time)

        if changed:
            ps.remove_all_structures()
            frame_idx = int(time/dt)
            time = frame_idx*dt
            ps_cloud = ps.register_point_cloud("my points", trajectory_xs[frame_idx])
            axis_length = 2.0
            x_axis = np.array([[0,0,0], [axis_length,0,0]])
            y_axis = np.array([[0,0,0], [0,axis_length,0]])
            z_axis = np.array([[0,0,0], [0,0,axis_length]])

            ps_cloud_x0 = ps.register_point_cloud("x0", x0[0])
            ps_cloud_x0.set_material("wax")
            ps_cloud_x0.set_radius(0.005)
            ps_cloud_x0.set_color((1.0,0.3,0.3))
            ps_cloud_x0.set_transparency(0.5)

            ps_cloud_xT = ps.register_point_cloud("xT", xT[0])
            ps_cloud_xT.set_material("wax")
            ps_cloud_xT.set_radius(0.005)
            ps_cloud_xT.set_color((0.3,1.0,0.3))
            ps_cloud_xT.set_transparency(0.5)

            ps.register_curve_network("x-axis", np.array([[0,0,0], [axis_length,0,0]]), np.array([[0,1]]))
            ps.register_curve_network("y-axis", np.array([[0,0,0], [0,axis_length,0]]), np.array([[0,1]]))
            ps.register_curve_network("z-axis", np.array([[0,0,0], [0,0,axis_length]]), np.array([[0,1]]))

            # Set axis colors
            ps.get_curve_network("x-axis").set_color((1,0,0))  # Red for X
            ps.get_curve_network("y-axis").set_color((0,1,0))  # Green for Y
            ps.get_curve_network("z-axis").set_color((0,0,1))  # Blue for Z
            plot_trajectory_3d_polyscope(trajectory_xs, frame_idx, "reverse_trajectory", simplified=False)

    ps.set_user_callback(imgui_callback)
    ps.show()
