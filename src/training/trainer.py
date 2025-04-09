import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import abc
from jax.lax import scan
import src.training.loss as Losses
from src.stochastics.sde import *
from src.stochastics.sde_solver import *
from typing import Optional
import jax.random as jrandom
from src.dataGenerator.spherical_data_generator import S2ManifoldDataGenerator, DataGenerator
from src.utils.KeyMonitor import KeyMonitor
from tqdm import tqdm
from functools import partial
from tqdm import trange
from jax import debug
class Trainer(abc.ABC):
    model: nn.Module
    @abc.abstractmethod
    def train_state_init(self, model: nn.Module, lr: float = 1e-3, model_kwargs: dict = {}):
        pass

    @abc.abstractmethod
    def train_epoch(self, train_state: train_state.TrainState, 
                   data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int):
        pass

    @abc.abstractmethod
    def train(self, train_state: train_state.TrainState, sde: SDE, solver: SDESolver, 
             data_generator: DataGenerator, epochs: int, batch_size: int):
        pass

class SsmTrainer(Trainer):
    def __init__(self, seed: int = 0, landmark_num: int = 32):
        self.key_monitor = KeyMonitor(seed)
        self.object_fn = "Heng"
        self.landmark_num = landmark_num
    def train_state_init(self, model: nn.Module, lr: float = 1e-3, model_kwargs: dict = {}, retrain: bool = False, ckpt_params: Optional[jnp.ndarray] = None):
        init_key = self.key_monitor.next_key()
        params = model.init(init_key, model_kwargs['x'], model_kwargs['t'], model_kwargs['x0'])
        if retrain:
            params = ckpt_params
        if 'object_fn' in model_kwargs:
            self.object_fn = model_kwargs['object_fn']
        tx = optax.adam(lr)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def _generate_batch(self, data_generator: DataGenerator, landmark_num: int, batch_size: int):
        return data_generator.generate_data(landmark_num, batch_size)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def _train_step(self, train_state: train_state.TrainState, x0: jnp.ndarray, sde: SDE, solver: SDESolver, solve_keys: jnp.ndarray):
        # Create new solver instance with provided key
        solver = solver.from_sde(
            sde=sde,
            dt=solver.dt,
            total_time=solver.total_time,
            dim=x0.shape[2],
        )
        
        # Solve SDE for each sample with provided keys
        training_data, diffusion_history = jax.vmap(solver.solve, in_axes=(0, 0))(x0, solve_keys)
    
        
        # Process data - make sure xs and times have matching dimensions
        num_timesteps = training_data.shape[1]
        times = jnp.linspace(0, solver.total_time, num_timesteps)
        print(times.shape)
        xs = training_data
        
        # Compute Sigmas and drifts for all timesteps
        Sigmas = jax.vmap(jax.vmap(sde.Sigma, in_axes=(0, 0)), in_axes=(0, None))(xs, times)
        drifts = jax.vmap(jax.vmap(sde.drift_fn, in_axes=(0, 0)), in_axes=(0, None))(xs, times)

        def loss_fn(params):
            loss = Losses.ssm_dsm_loss(params, train_state, xs, times, x0, Sigmas, drifts, object_fn=self.object_fn)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    def train_epoch(self, train_state: train_state.TrainState, 
                   data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int):
        x0 = self._generate_batch(data_generator, self.landmark_num, batch_size)
        solve_keys = self.key_monitor.split_keys(x0.shape[0])
        return self._train_step(train_state, x0, sde, solver, solve_keys)

    def train(self, train_state: train_state.TrainState, sde: SDE, solver: SDESolver, 
             data_generator: DataGenerator, epochs: int, batch_size: int):
        losses = jnp.zeros(epochs)
        # print the loss in the tqdm progress bar
        t = trange(epochs, desc="Bar desc")
        for i in t:
            train_state, loss = self.train_epoch(train_state, data_generator, sde, solver, batch_size)
            losses = losses.at[i].set(loss)
            t.set_description(f"Training loss: {loss}")
            t.refresh()
        return train_state, losses


class NeuralOpTrainer(Trainer):
    def __init__(self, seed: int = 0, landmark_num: int = 32):
        self.key_monitor = KeyMonitor(seed)
        self.object_fn = "Heng"
        self.landmark_num = landmark_num

    def train_state_init(self, model: nn.Module, lr: float = 1e-3, model_kwargs: dict = {}, retrain: bool = False, ckpt_params: Optional[jnp.ndarray] = None):
        """Initialize training state for neural operator model
        
        Args:
            model: Neural operator model (CTUNO1D or CTUNO2D)
            lr: Learning rate for optimizer
            model_kwargs: Dictionary containing:
                - x: Input data tensor
                - t: Time points tensor
                - object_fn: Optional loss function name
        """
        init_key = self.key_monitor.next_key()
        
        # Initialize model parameters - removed train parameter
        params = model.init(init_key, model_kwargs['x'], model_kwargs['t'], model_kwargs['x_L'])
        if retrain:
            params = ckpt_params
        
        # Set object function if provided
        if 'object_fn' in model_kwargs:
            self.object_fn = model_kwargs['object_fn']
            
        # Initialize optimizer
        tx = optax.adam(lr)
        
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )

    def _generate_batch(self, data_generator: DataGenerator, landmark_num: int, batch_size: int):
        return data_generator.generate_data(landmark_num, batch_size)

    @partial(jax.jit, static_argnums=(0, 3, 4, 6))
    def _train_step(self, train_state: train_state.TrainState, x0: jnp.ndarray, sde: SDE, solver: SDESolver, solve_keys: jnp.ndarray, x_L: int):
        # Create new solver instance with provided key
        solver = solver.from_sde(
            sde=sde,
            dt=solver.dt,
            total_time=solver.total_time,
            dim=x0.shape[-1],
        )
        
        # Solve SDE for each sample with provided keys
        training_data, diffusion_history = jax.vmap(solver.solve, in_axes=(0, 0))(x0, solve_keys)
        
        # Process data
        num_timesteps = training_data.shape[1]
        times = jnp.linspace(0, solver.total_time, num_timesteps)
        xs = training_data
        
        # Compute Sigmas and drifts for all timesteps
        Sigmas = jax.vmap(jax.vmap(sde.Sigma, in_axes=(0, 0)), in_axes=(0, None))(xs, times)
        drifts = jax.vmap(jax.vmap(sde.drift_fn, in_axes=(0, 0)), in_axes=(0, None))(xs, times)

        def loss_fn(params):
            # Neural operator specific loss function
            loss = Losses.ssm_dsm_loss(params, train_state, xs, times, x0, Sigmas, drifts, object_fn=self.object_fn, with_x0=False, x_L=x_L)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        # grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        # debug.print(f"Gradient norm: {grad_norm}")
        # for layer_name, layer_params in jax.tree_util.tree_leaves_with_path(grads):
        #     param_path = '/'.join(str(p) for p in layer_name)
            
        #     # 如果是叶子节点（实际参数）
        #     if not isinstance(layer_params, dict):
        #         grad_norm = jnp.linalg.norm(layer_params)
        #         grad_mean = jnp.mean(layer_params)
        #         grad_max = jnp.max(jnp.abs(layer_params))
        #         nan_check_arrays = [jnp.isnan(g) | jnp.isinf(g) for g in layer_params]
        #         has_nan_or_inf = jnp.any(jnp.stack(nan_check_arrays))
        #         debug.print(
        #             "Layer: {path}, Shape: {shape}, Norm: {norm}, Mean: {mean}, Max: {max}, Has NaN or Inf: {has_nan_or_inf}", 
        #             path=param_path,
        #             shape=layer_params.shape, 
        #             norm=grad_norm, 
        #             mean=grad_mean,
        #             max=grad_max,
        #             has_nan_or_inf=has_nan_or_inf
        #         )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    def train_epoch(self, train_state: train_state.TrainState, 
                   data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int, x_L: int):
        x0 = self._generate_batch(data_generator, self.landmark_num, batch_size)
        solve_keys = self.key_monitor.split_keys(x0.shape[0])
        return self._train_step(train_state, x0, sde, solver, solve_keys, x_L)

    def train(self, train_state: train_state.TrainState, sde: SDE, solver: SDESolver, 
              data_generator: DataGenerator, epochs: int, batch_size: int, x_L: int):
        losses = jnp.zeros(epochs)
        # print the loss in the tqdm progress bar
        t = trange(epochs, desc="Training neural operator")
        for i in t:
            train_state, loss = self.train_epoch(train_state, data_generator, sde, solver, batch_size, x_L)
            losses = losses.at[i].set(loss)
            t.set_description(f"Training loss: {loss}")
            t.refresh()
        return train_state, losses
    