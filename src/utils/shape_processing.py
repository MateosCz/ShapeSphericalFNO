import jax.numpy as jnp

def radial_projection(points, eps=1e-6):
    '''
    radial projection on unit sphere
    '''
    norms = jnp.linalg.norm(points, axis=-1, keepdims=True) + eps
    projected_points = points / norms
    return projected_points
