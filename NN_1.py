import jax.numpy as jnp
from flax import linen as nn


class ANN_64_32_16(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        
        x = nn.Dense(32)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        x = nn.Dense(16)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        
        return x.squeeze(-1)

class ANN_64_64_32(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        x = nn.Dense(32)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)

        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        
        return x.squeeze(-1)