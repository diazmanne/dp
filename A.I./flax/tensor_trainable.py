import jax
import jax.numpy as jnp
from flax import linen as nn

# Define the trainable tensor inside a simple model
class SimpleModel(nn.Module):
    def setup(self):
        self.tensor = self.param('tensor', lambda rng: jax.random.normal(rng, (2, 1)))

    def __call__(self, x):
        return jnp.dot(x, self.tensor)

# Instantiate the model
model = SimpleModel()

# Create a dummy input for initialization
dummy_input = jnp.ones((1, 2))  # Shape (1,2) to match tensor shape (2,1)

# Initialize model parameters with a PRNG key and input
params = model.init(jax.random.PRNGKey(0), dummy_input)

# Access the trainable parameter
tensor_train = params['params']['tensor']

print("Trainable parameter:", tensor_train)