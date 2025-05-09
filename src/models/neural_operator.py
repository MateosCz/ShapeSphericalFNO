import jax
import jax.numpy as jnp
from flax import linen as nn
import s2fft
from src.models.blocks import *
from src.utils.sht_helper import *
class CTShapeSFNO(nn.Module):
    '''
    Continuous Time Shape SFNO(Spherical Fourier Neural Operator),
    a U-shaped neural operator, with a list of downsampling and upsampling CTSFNOBlocks
    initialize parameters:
        l_list: tuple, the list of L (the maximum degree of the spherical harmonics) 
            for the downsampling and upsampling CTSFNOBlocks.(grid resolution).
        lift_dim: int, the dimension of the lifted feature
        latent_feature_dims: tuple, the list of feature dimensions for the downsampling and upsampling CTSFNOBlocks
        sampling: str, the sampling method
        activation: str, the activation function

    The l_list and the latent_feature_dims should have the same length.
    '''
    x_feature_dim: int
    l_list: tuple
    lift_dim: int
    latent_feature_dims: tuple
    sampling: str = "mw"
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x, t, x_L):
        time_emb_dim = self.lift_dim * 4
        # time_emb = TimeEmbedding(t_emb_dim=time_emb_dim)(t)
        time_emb_phi = TimeEmbedding(c=time_emb_dim)(t)
        time_emb_psi = TimeEmbedding(c=time_emb_dim)(t)
        flatten = False
        phi_dim = get_phi_dim(x_L, self.sampling)

        # if x is flattened, then we need to reshape it to 2D manifold data
        if x.ndim == 2:
            if self.sampling == "dh":
                x = jnp.reshape(x, (2*x_L, phi_dim, x.shape[1]))
            elif self.sampling == "mw":
                x = jnp.reshape(x, (x_L, phi_dim, x.shape[1]))
            flatten = True

        x = nn.Dense(features=self.lift_dim, kernel_init=nn.initializers.normal(0.1))(x)
        x = nn.gelu(x)
        final_l_spatial = infer_L_from_shape(x, self.sampling)

        l_list = self.l_list

        latent_feature_dims = (1,) + self.latent_feature_dims

        down_list = []


        

        # downsampling
        for i in range(len(l_list)):
            x = CTSFNOBlock(L_freq_used=l_list[i], 
                            L_out_spatial=l_list[i],
                            in_channels=latent_feature_dims[i] * self.lift_dim, 
                            out_channels=latent_feature_dims[i+1] * self.lift_dim, 
                            path="down",
                            sampling=self.sampling, 
                            activation=self.activation,
                            grid_embedding=True)(x, time_emb_phi, time_emb_psi)
            down_list.append(x)

        # lowest block
        x = CTSFNOBlock(L_freq_used=l_list[-1], 
                        L_out_spatial=l_list[-1],
                        in_channels=latent_feature_dims[-1] * self.lift_dim, 
                        out_channels=latent_feature_dims[-1] * self.lift_dim, 
                        path="middle",
                        sampling=self.sampling, 
                        activation=self.activation,
                        grid_embedding=True)(x, time_emb_phi, time_emb_psi)
                        

        # upsampling

        for i in range(1, len(l_list)):
            x = jnp.concatenate([x, down_list[-(i)]], axis=-1)
            x = CTSFNOBlock(L_freq_used=l_list[-(i+1)], 
                            L_out_spatial=l_list[-(i+1)],
                            in_channels=latent_feature_dims[-(i)] * self.lift_dim * 2, 
                            out_channels=latent_feature_dims[-(i+1)] * self.lift_dim, 
                            path="up",
                            sampling=self.sampling, 
                            activation=self.activation,
                            grid_embedding=True)(x, time_emb_phi, time_emb_psi)
        # outermost block
        x = jnp.concatenate([x, down_list[0]], axis=-1)
        x = CTSFNOBlock(L_freq_used=l_list[0],
                        L_out_spatial=final_l_spatial,
                        in_channels=latent_feature_dims[1] * self.lift_dim * 2, 
                        out_channels= latent_feature_dims[0] * self.lift_dim, 
                        path="up",
                        sampling=self.sampling, 
                        activation=self.activation)(x, time_emb_phi, time_emb_psi)
        # x = pad_inverse_output(x, self.sampling)
        # x = nn.Conv(features=self.x_feature_dim, kernel_size=(1, 1), padding="VALID")(x)
        x = nn.Dense(features=self.x_feature_dim, kernel_init=nn.initializers.normal(0.1))(x)
        if flatten:
            x = jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        return x
    
class CTSFNO(nn.Module):
    '''
    Continuous Time SFNO(Spherical Fourier Neural Operator),without the rescaling the grid resolution
    A time dependent operator based on the Part et al. Learning PDE Solution Operator for Continuous Modeling of Time-Series
    '''

