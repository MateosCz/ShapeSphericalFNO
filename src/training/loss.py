import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(7, 8, 9))
def ssm_dsm_loss(params, state, xs, times, x0, Sigmas, drifts, object_fn='Heng', with_x0=True, x_L=12):
    dt = times[1] - times[0]
    # dimensions:
    # Sigmas: (batch_size, num_timesteps, num_landmarks, dim)
    # xs: (batch_size, num_timesteps, num_landmarks, dim)
    # times: (num_timesteps,)
    # x0: (batch_size, num_landmarks, dim)
    # drifts: (batch_size, num_timesteps, num_landmarks, dim)

    # vmap over timesteps first then over batch size
    # the inner vmap is over the batch size
    loss = jax.vmap(batched_single_step_loss, in_axes=(None, #params
                                                        None, #state
                                                        1, #x_prev
                                                        1, #x
                                                        0, #t
                                                        None, #x0
                                                        1, #Sigma
                                                        1, #Sigma_prev
                                                        1, #drift_prev
                                                        None, #dt
                                                        None, #object_fn
                                                        None, #with_x0
                                                        None, #x_L
                                                        ))(params, 
                                                               state, 
                                                               xs[:, :-1, ...], 
                                                               xs[:, 1:, ...], 
                                                               times[:-1], 
                                                               x0, 
                                                               Sigmas[:, 1:, ...], 
                                                               Sigmas[:, :-1, ...], 
                                                               drifts[:, :-1, ...], 
                                                               dt, 
                                                               object_fn,
                                                               with_x0,
                                                               x_L)
    
    print(loss.shape)
    if object_fn == 'Heng':
        loss = jnp.mean(loss, axis=1)
        loss = jnp.sum(loss) * dt/2
    elif object_fn == 'Novel':
        loss = jnp.sum(loss)/xs.shape[0]
        loss = jnp.sum(loss)/2
    elif object_fn == 'Yang':
        loss = jnp.mean(loss, axis=1)
        loss = jnp.sum(loss) * dt/2

    return loss
        
def single_step_loss(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn='Heng', with_x0=True, x_L=12):
    print("x.shape", x.shape)
    print("x_prev.shape", x_prev.shape)
    print("drift_prev.shape", drift_prev.shape)
    print("dt.shape", dt.shape)
        
    if object_fn == 'Heng':
        if with_x0:
            pred_score = state.apply_fn(params, x, t, x0, x_L)
        else:
            pred_score = state.apply_fn(params, x, t, x_L)

        # check the x's shape, if it is a 2D manifold data, then we need to flatten it
        if x.ndim == 3:
            x = jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
            x_prev = jnp.reshape(x_prev, (x_prev.shape[0] * x_prev.shape[1], x_prev.shape[2]))
            pred_score = jnp.reshape(pred_score, (pred_score.shape[0] * pred_score.shape[1], pred_score.shape[2]))
            drift_prev = jnp.reshape(drift_prev, (drift_prev.shape[0] * drift_prev.shape[1], drift_prev.shape[2]))
            Sigma_prev = jnp.reshape(Sigma_prev, (Sigma_prev.shape[0] * Sigma_prev.shape[1], Sigma_prev.shape[2] * Sigma_prev.shape[3]))
            # Sigma_original = Sigma
            Sigma = jnp.reshape(Sigma, (Sigma.shape[0] * Sigma.shape[1], Sigma.shape[2] * Sigma.shape[3]))
            # diff_Sigma = Sigma - Sigma_original
            # largest_diff = jnp.max(jnp.abs(diff_Sigma))
            # print("largest_diff", largest_diff)
            
            # Add regularization as in notebook
            # reg_factor = jax.lax.cond(jnp.linalg.norm(Sigma_prev) > 1e-3, lambda: 1e-2, lambda: 1e-3)
            # Sigma_prev = Sigma_prev + reg_factor * jnp.eye(Sigma_prev.shape[0])
            Sigma_prev = Sigma_prev + 1e-3 * jnp.eye(Sigma_prev.shape[0])
            Sigma_prev_inv = jnp.linalg.solve(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))
            # Sigma_prev_inv = jnp.linalg.lstsq(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))[0]
            # Sigma_prev_inv = jnp.linalg.pinv(Sigma_prev,rcond=1e-6)
            # Sigma_prev_inv = jnp.linalg.inv(Sigma_prev)
            g_approx = -jnp.matmul(Sigma_prev_inv, (x - x_prev - dt * drift_prev))/dt
            
            diff = pred_score - g_approx
            # check the x's shape
            loss = jnp.linalg.norm(jnp.matmul(diff.T, jnp.matmul(Sigma * dt, diff))) ** 2
            return loss
        # Add regularization as in notebook

        Sigma_prev = Sigma_prev + 1e-3 * jnp.eye(Sigma_prev.shape[0])
        # Sigma_prev_inv = jnp.linalg.lstsq(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))[0]
        Sigma_prev_inv = jnp.linalg.solve(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))
        # Sigma_prev_inv = jnp.linalg.pinv(Sigma_prev)
        g_approx = -jnp.matmul(Sigma_prev_inv, (x - x_prev - dt * drift_prev))/dt
        
        diff = pred_score - g_approx
        # check the x's shape
        loss = jnp.linalg.norm(jnp.matmul(diff.T, jnp.matmul(Sigma * dt, diff))) ** 2
    elif object_fn == 'Novel':
        # Novel version from notebook
        pred_score = state.apply_fn(params, x_prev, t-dt, x0)
        approx_stable = (x - x_prev - dt * drift_prev)
        loss = pred_score.T @ (Sigma_prev * dt) @ pred_score + 2 * pred_score.T @ approx_stable
        loss = loss * dt 
    elif object_fn == 'Yang':
        if with_x0:
            pred_score = state.apply_fn(params, x, t, x0, x_L)
            # pred_score = state.apply_fn(params, x, t, x0)
        else:
            pred_score = state.apply_fn(params, x, t, x_L)
            # pred_score = state.apply_fn(params, x, t)
        
        # add penalty on polar parts
        # theta_grid = jnp.arccos(x[..., 2])
        # polar_mask = (theta_grid < 0.2) | (theta_grid > jnp.pi - 0.2)
        # polar_penalty = jnp.mean(jnp.linalg.norm(pred_score, axis=-1) * polar_mask)
        b = -(x - x_prev - dt * drift_prev) / dt
        # loss_main = jnp.linalg.norm(pred_score - b) ** 2 + 5e-3 * polar_penalty
        # terminal_mask = jnp.isclose(t, 1.0, atol=1e-2)
        # attractor = (x-x0) / 1e-2
        # penalty = jnp.mean(jnp.linalg.norm(pred_score + attractor, axis=-1) ** 2) * terminal_mask
        # loss = jnp.linalg.norm(pred_score - b, axis=-1) ** 2
        loss = jnp.mean(jnp.sum(jnp.square(pred_score - b), axis=-1),axis=-1)

        # delta = 0.2
        # theta_grid = jnp.arccos(x[..., 2])
        # equator_mask = (theta_grid > jnp.pi / 2 - delta) & (theta_grid < jnp.pi / 2 + delta)
        # equator_penalty = jnp.mean(jnp.linalg.norm(pred_score, axis=-1) * equator_mask)
        # loss += 1e-3 * equator_penalty


        
        # loss = jax.lax.cond(
        #     terminal_mask,
        #     lambda _: loss_main + 1.0 * penalty,
        #     lambda _: loss_main,
        #     operand=None
        # )
            
    return loss
# vmap over batch size, one batch's loss is mean at each timestep's loss
def batched_single_step_loss(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn='Heng', with_x0=True, x_L=12):
    batched_loss = jax.vmap(single_step_loss, in_axes=(None, #params
                                                        None, #state
                                                        0, #x_prev
                                                        0, #x
                                                        None, #t
                                                        0, #x0
                                                        0, #Sigma
                                                        0, #Sigma_prev
                                                        0, #drift_prev
                                                        None, #dt
                                                        None, #object_fn
                                                        None, #with_x0
                                                        None, #x_L
                                                        ))(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn, with_x0, x_L)
    return batched_loss

