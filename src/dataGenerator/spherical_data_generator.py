import jax
import jax.numpy as jnp
import jax.random as jrandom
import abc

class DataGenerator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_data(self, key: jnp.ndarray, batch_size: int):
        pass


# spherical coordinate sampled data generator 
# all the data with shape (L, 2L-1, N) is sampled on the unit sphere with the feature channels N
# in this case, the feature channels are decart coordinates
class S2ManifoldDataGenerator(DataGenerator):
    """
    Generate data on various manifolds with s2fft-compatible sampling.
    
    This class can generate data on different manifolds (sphere, cylinder, torus)
    with sampling schemes compatible with s2fft for spherical harmonic transforms.
    """
    
    def __init__(self, sampling: str = "mw", manifold_type: str = "sphere", 
                 radius: float = 1.0, height: float = 2.0, minor_radius: float = 0.5, 
                 major_radius: float = 2.0, width: float = 0.5, center: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]), flatten: bool = False, seed=0, randomization: bool = False):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.key = jax.random.PRNGKey(seed)
        self.sampling = sampling
        self.manifold_type = manifold_type
        self.radius = radius
        self.height = height
        self.minor_radius = minor_radius
        self.major_radius = major_radius
        self.width = width
        self.center = center
        self.flatten = flatten
        self.randomization = randomization
    def generate_sampling_grid(self, L, sampling='mw'):
        """
        Generate angular sampling grid based on the requested scheme.
        
        Args:
            L: Bandwidth/resolution parameter
            sampling: Sampling scheme ('mw', 'mwss', or 'dh')
            
        Returns:
            theta_grid: 2D grid of theta values
            phi_grid: 2D grid of phi values
            ntheta: Number of theta samples
            nphi: Number of phi samples
        """
        if self.randomization:
            self.key, key = jrandom.split(self.key)
        
        if sampling == 'mw':
            # McEwen & Wiaux sampling
            ntheta = L
            nphi = 2*L-1
            
            if self.randomization:
                theta = jrandom.uniform(key, (ntheta,), minval=0, maxval=jnp.pi)
                phi = jrandom.uniform(key, (nphi,), minval=0, maxval=2*jnp.pi)
            else:
                theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=True)
                phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
            
        elif sampling == 'mwss':
            # McEwen & Wiaux Symmetric Sampling
            ntheta = L + 1
            nphi = 2*L
            if self.randomization:
                theta = jrandom.uniform(key, (ntheta,), minval=0, maxval=jnp.pi)
                phi = jrandom.uniform(key, (nphi,), minval=0, maxval=2*jnp.pi)
            else:
                theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=True)
                phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
            
        elif sampling == 'dh':
            # Driscoll & Healy sampling
            ntheta = 2*L
            nphi = 2*L
            if self.randomization:
                theta = jrandom.uniform(key, (ntheta,), minval=0, maxval=jnp.pi) + jnp.pi/(2 *ntheta)
                phi = jrandom.uniform(key, (nphi,), minval=0, maxval=2*jnp.pi)
            else:
                theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=False) + jnp.pi/(2 *ntheta)
                phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
            # drop the last phi
            phi = phi[:-1]
            nphi = nphi - 1
            
        else:
            raise ValueError(f"Unsupported sampling scheme: {sampling}. Use 'mw', 'mwss', or 'dh'")
        
        # Create meshgrid
        phi_grid, theta_grid = jnp.meshgrid(phi, theta)
        
        return theta_grid, phi_grid, ntheta, nphi
    
    def sphere(self, theta_grid, phi_grid, radius=1.0, center=None):
        """
        Generate points on a sphere.
        
        Args:
            theta_grid: Grid of theta values
            phi_grid: Grid of phi values
            radius: Sphere radius
            center: Sphere center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the sphere
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Convert spherical to Cartesian coordinates
        x = radius * jnp.sin(theta_grid) * jnp.cos(phi_grid)
        y = radius * jnp.sin(theta_grid) * jnp.sin(phi_grid)
        z = radius * jnp.cos(theta_grid)
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    def cylinder(self, theta_grid, phi_grid, radius=1.0, height=2.0, center=None):
        """
        Generate points on a cylinder.
        
        Args:
            theta_grid: Grid of theta values (used as height parameter)
            phi_grid: Grid of phi values (used as angular parameter)
            radius: Cylinder radius
            height: Cylinder height
            center: Cylinder center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the cylinder
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Map theta from [0, pi] to [-height/2, height/2]
        h = height * (theta_grid / jnp.pi - 0.5)
        
        # Convert cylindrical to Cartesian coordinates
        x = radius * jnp.cos(phi_grid)
        y = radius * jnp.sin(phi_grid)
        z = h
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    def torus(self, theta_grid, phi_grid, major_radius=2.0, minor_radius=0.5, center=None):
        """
        Generate points on a torus.
        
        Args:
            theta_grid: Grid of theta values (poloidal angle)
            phi_grid: Grid of phi values (toroidal angle)
            major_radius: Distance from center of tube to center of torus
            minor_radius: Radius of the tube
            center: Torus center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the torus
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Convert torus parameters to Cartesian coordinates
        x = (major_radius + minor_radius * jnp.cos(theta_grid)) * jnp.cos(phi_grid)
        y = (major_radius + minor_radius * jnp.cos(theta_grid)) * jnp.sin(phi_grid)
        z = minor_radius * jnp.sin(theta_grid)
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    def mobius_strip(self, theta_grid, phi_grid, radius=2.0, width=0.5, center=None):
        """
        Generate points on a Möbius strip.
        
        Args:
            theta_grid: Grid of theta values
            phi_grid: Grid of phi values
            radius: Main radius of the Möbius strip
            width: Width of the strip
            center: Center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the Möbius strip
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Remap theta to [0, 2π]
        u = phi_grid
        # Remap phi to [-width/2, width/2]
        v = width * (theta_grid / jnp.pi - 0.5)
        
        # Parametric equations for Möbius strip
        x = (radius + v * jnp.cos(u/2)) * jnp.cos(u)
        y = (radius + v * jnp.cos(u/2)) * jnp.sin(u)
        z = v * jnp.sin(u/2)
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    def fib_sphere(self, theta_grid, phi_grid, radius=1.0, center=None):
        """
        Generate points on a Fibonacci sphere.
        
        Args:
            theta_grid: Grid of theta values
            phi_grid: Grid of phi values
            radius: Sphere radius
            center: Sphere center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the Fibonacci sphere
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        n_points = theta_grid.shape[0] * theta_grid.shape[1]
        golden_ratio = (1 + jnp.sqrt(5)) / 2
        golden_angle = jnp.pi * (3 - jnp.sqrt(5))
        
        indices = jnp.arange(n_points)
        z = 1 - (2 * indices + 1) / n_points
        theta = jnp.arccos(z)
        phi = golden_angle * indices
        
        # Convert spherical to Cartesian coordinates
        x = radius * jnp.sin(theta) * jnp.cos(phi)
        y = radius * jnp.sin(theta) * jnp.sin(phi)
        z = radius * jnp.cos(theta)
        
        # Apply center offset   
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)

        n_L = theta_grid.shape[0]
        n_m = theta_grid.shape[1]
        points = jnp.reshape(points, (n_L, n_m, 3))
        
        return points
    
    def _generate_data_single(self, L, sampling, key):
        theta_grid, phi_grid, ntheta, nphi = self.generate_sampling_grid(L, sampling)
        if self.manifold_type == 'sphere':
            points = self.sphere(theta_grid, phi_grid, self.radius, self.center)
        elif self.manifold_type == 'cylinder':
            points = self.cylinder(theta_grid, phi_grid, self.radius, self.height, self.center)
        elif self.manifold_type == 'torus':
            points = self.torus(theta_grid, phi_grid, self.major_radius, self.minor_radius, self.center)
        elif self.manifold_type == 'mobius':
            points = self.mobius_strip(theta_grid, phi_grid, self.radius, self.width, self.center)
        elif self.manifold_type == 'fib_sphere':
            points = self.fib_sphere(theta_grid, phi_grid, self.radius, self.center)
        else:
            raise ValueError(f"Unsupported manifold type: {self.manifold_type}")
            
        return points

        
    
    def generate_data(self, L, batch_size=1, **kwargs):
        """
        Generate data on the specified manifold with s2fft-compatible sampling.
        
        Args:
            L: Bandwidth/resolution parameter
            manifold_type: Type of manifold ('sphere', 'cylinder', 'torus', 'mobius')
            sampling: Sampling scheme ('MW', 'MWSS', or 'DH')
            batch_size: Batch size
            **kwargs: Additional parameters for the specific manifold
                - radius, center for sphere
                - radius, height, center for cylinder
                - major_radius, minor_radius, center for torus
                - radius, width, center for mobius_strip
            
        Returns:
            points: Tensor of shape (batch_size, ntheta, nphi, 3) with 3D coordinates
        """
        # Generate appropriate sampling grid
        # theta_grid, phi_grid, ntheta, nphi = self.generate_sampling_grid(L, self.sampling)


        # Generate points on the requested manifold
        # if self.manifold_type == 'sphere':
        #     points = self.sphere(theta_grid, phi_grid, self.radius, self.center)
        # elif self.manifold_type == 'cylinder':
        #     points = self.cylinder(theta_grid, phi_grid, self.radius, self.height, self.center)
        # elif self.manifold_type == 'torus':
        #     points = self.torus(theta_grid, phi_grid, self.major_radius, self.minor_radius, self.center)
        # elif self.manifold_type == 'mobius':
        #     points = self.mobius_strip(theta_grid, phi_grid, self.radius, self.width, self.center)
        # elif self.manifold_type == 'fib_sphere':
        #     points = self.fib_sphere(theta_grid, phi_grid, self.radius, self.center)
        # else:
        #     raise ValueError(f"Unsupported manifold type: {self.manifold_type}")
        if batch_size > 1:
            self.key, key_new = jrandom.split(self.key)
            key_new = jrandom.split(key_new, batch_size)
            points = jax.vmap(self._generate_data_single, in_axes=(None, None, 0))(L, self.sampling, key_new)
            # print(points.shape)
        else:
            points = self._generate_data_single(L, self.sampling, self.key)
            points = points[None, ...]
            # print(points.shape)
        
        # # Add batch dimension
        # if batch_size > 1:
        #     points = jnp.tile(points[None, ...], (batch_size, 1, 1, 1))
        # else:
        #     points = points[None, ...]
        if self.flatten:
            points = jnp.reshape(points, (points.shape[0], points.shape[1] * points.shape[2], points.shape[3]))
            
        return points