import jax
import jax.numpy as jnp
import jax.random as jrandom
import abc
import numpy as np
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
                 major_radius: float = 2.0, width: float = 0.5, center: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]), flatten: bool = False, seed=0, randomization: bool = False, epsilon1: float = 1.0, epsilon2: float = 1.0, a: float = 1.0, b: float = 1.0, c: float = 1.0, 
                 A: float = 0.3, n: int = 4, m: int = 5):
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
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.a = a
        self.b = b
        self.c = c  
        self.A = A
        self.n = n
        self.m = m
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
        elif sampling == 'gl':
            
            ntheta = L
            nphi = 2 * L - 1
            nodes, weights = np.polynomial.legendre.leggauss(L)
            if self.manifold_type == 'torus':
                theta = jnp.linspace(0, 2*jnp.pi, ntheta, endpoint=False)
            else:
                theta = jnp.flip(jnp.arccos(nodes))
            phi = jnp.linspace(0, 2 * jnp.pi, nphi, endpoint=False)
            
        else:
            raise ValueError(f"Unsupported sampling scheme: {sampling}. Use 'mw', 'mwss', 'dh', or 'gl'")
        
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
    
    def torus(self, theta_grid, phi_grid, major_radius=0.8, minor_radius=0.2, center=None):
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
    
    def closed_cylinder(self, theta_grid, phi_grid, radius=1.0, height=2.0, center=None):
        """
        Generate a closed cylinder (with side + top/bottom caps).

        theta_grid: vertical (z-axis) index → Gauss-Legendre
        phi_grid: angular around z-axis
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])

        # Side surface
        x = radius * jnp.cos(phi_grid)
        y = radius * jnp.sin(phi_grid)
        z = height * (theta_grid / jnp.pi - 0.5)  # GL: theta ∈ [0, π]

        side_points = jnp.stack([x, y, z], axis=-1)

        # Bottom cap
        mask_bottom = theta_grid < 0.25 * jnp.pi
        r_bottom = radius * (theta_grid / (0.25 * jnp.pi))
        cap_x = r_bottom * jnp.cos(phi_grid)
        cap_y = r_bottom * jnp.sin(phi_grid)
        cap_z = jnp.full_like(cap_x, -height / 2)
        bottom_points = jnp.stack([cap_x, cap_y, cap_z], axis=-1)

        # Top cap
        mask_top = theta_grid > 0.75 * jnp.pi
        r_top = radius * ((jnp.pi - theta_grid) / (0.25 * jnp.pi))
        cap_x = r_top * jnp.cos(phi_grid)
        cap_y = r_top * jnp.sin(phi_grid)
        cap_z = jnp.full_like(cap_x, height / 2)
        top_points = jnp.stack([cap_x, cap_y, cap_z], axis=-1)

        # 替换边界点
        points = jnp.where(mask_bottom[..., None], bottom_points, side_points)
        points = jnp.where(mask_top[..., None], top_points, points)

        # Apply center shift
        points = points + center[None, None, :]

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
    
    def heart_surface(self, theta_grid, phi_grid, center=None):
        """
        生成一个 3D 心形曲面点云，支持 (L, 2L-1, 3) 格式。
        theta: ∈ [0, π]
        phi: ∈ [0, 2π]
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])

        # 半径函数 r(θ)
        sin_t = jnp.sin(theta_grid)
        cos_t = jnp.cos(theta_grid)
        r = 2 - 2 * sin_t + (sin_t * jnp.sqrt(jnp.abs(cos_t))) / (sin_t + 1.4)

        # 转换为 3D 坐标
        x = r * sin_t * jnp.cos(phi_grid)
        y = r * sin_t * jnp.sin(phi_grid)
        z = r * cos_t

        points = jnp.stack([x, y, z], axis=-1)
        points = points + center[None, None, :]
        return points
    
    def superquadric_sphere(self, theta_grid, phi_grid, epsilon1=1.0, epsilon2=1.0, a=1.0, center=None):
        """
        生成超二次球面（superquadric sphere）点云。
        θ ∈ [0, π], φ ∈ [0, 2π]
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])

        # helper: 处理负数指数时的 sign 保留
        def sgnpow(x, p):
            return jnp.sign(x) * jnp.abs(x) ** p

        # 计算每个维度
        x = a * sgnpow(jnp.sin(theta_grid), epsilon1) * sgnpow(jnp.cos(phi_grid), epsilon2)
        y = a * sgnpow(jnp.sin(theta_grid), epsilon1) * sgnpow(jnp.sin(phi_grid), epsilon2)
        z = a * sgnpow(jnp.cos(theta_grid), epsilon1)

        pts = jnp.stack([x, y, z], axis=-1)
        pts = pts + center[None, None, :]
        return pts
    
    def superellipsoid(self, theta_grid, phi_grid, 
                    a=1.0, b=1.0, c=1.0, 
                    epsilon1=1.0, epsilon2=1.0, 
                    center=None):
        """
        生成一个 (L, 2L-1, 3) 格式的 super ellipsoid 点云
        参数:
            theta_grid: ∈ [0, π]
            phi_grid: ∈ [0, 2π]
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        def sgnpow(x, p):
            return jnp.sign(x) * jnp.abs(x) ** p

        cos_theta = jnp.cos(theta_grid)
        sin_theta = jnp.sin(theta_grid)
        cos_phi = jnp.cos(phi_grid)
        sin_phi = jnp.sin(phi_grid)

        x = a * sgnpow(cos_theta, epsilon1) * sgnpow(cos_phi, epsilon2)
        y = b * sgnpow(cos_theta, epsilon1) * sgnpow(sin_phi, epsilon2)
        z = c * sgnpow(sin_theta, epsilon1)

        pts = jnp.stack([x, y, z], axis=-1)
        pts = pts + center[None, None, :]
        return pts
    
    def bump_sphere(self, theta_grid, phi_grid, radius=1.0, A=0.3, n=4, m=5, center=None):
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])

        r = radius + A * jnp.sin(n * theta_grid) * jnp.cos(m * phi_grid)

        x = r * jnp.sin(theta_grid) * jnp.cos(phi_grid)
        y = r * jnp.sin(theta_grid) * jnp.sin(phi_grid)
        z = r * jnp.cos(theta_grid)

        pts = jnp.stack([x, y, z], axis=-1)
        pts = pts + center[None, None, :]
        return pts
    
    def _generate_data_single(self, L, sampling, key):
        theta_grid, phi_grid, ntheta, nphi = self.generate_sampling_grid(L, sampling)
        if self.manifold_type == 'sphere':
            points = self.sphere(theta_grid, phi_grid, self.radius, self.center)
        elif self.manifold_type == 'cylinder':
            points = self.cylinder(theta_grid, phi_grid, self.radius, self.height, self.center)
        elif self.manifold_type == 'torus':
            points = self.torus(theta_grid, phi_grid, self.major_radius, self.minor_radius, self.center)
        elif self.manifold_type == 'closed_cylinder':
            points = self.closed_cylinder(theta_grid, phi_grid, self.radius, self.height, self.center)
        elif self.manifold_type == 'mobius':
            points = self.mobius_strip(theta_grid, phi_grid, self.radius, self.width, self.center)
        elif self.manifold_type == 'fib_sphere':
            points = self.fib_sphere(theta_grid, phi_grid, self.radius, self.center)
        elif self.manifold_type == 'heart_surface':
            points = self.heart_surface(theta_grid, phi_grid, self.center)
        elif self.manifold_type == 'superquadric_sphere':
            points = self.superquadric_sphere(theta_grid, phi_grid, self.epsilon1, self.epsilon2, self.a, self.center)
        elif self.manifold_type == 'superellipsoid':
            points = self.superellipsoid(theta_grid, phi_grid, self.a, self.b, self.c, self.epsilon1, self.epsilon2, self.center)
        elif self.manifold_type == 'bump_sphere':
            points = self.bump_sphere(theta_grid, phi_grid, self.radius, self.A, self.n, self.m, self.center)
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