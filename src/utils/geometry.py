import jax.numpy as jnp
import jax
import jax.random as jr


def spherical_distance(theta1, phi1, theta2, phi2):
    """Calculate distance between two points on sphere"""
    # Using variant of haversine formula
    dtheta = theta2 - theta1
    dphi = phi2 - phi1
    a = jnp.sin(dtheta/2)**2 + jnp.sin(theta1) * jnp.sin(theta2) * jnp.sin(dphi/2)**2
    return 2 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0, 1)))



