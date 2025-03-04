import jax 
import jax.numpy as jnp 
from flax import linen as nn

#define the trainable tensor

class SimpleModel(nn.Model):
    def setup(self):
        self.tensor = self.param('tensor', lambda rgn: jax.random.normal(rgn,(2,1)))

    def __call__(self,x):
        return jnp.dot(x,self.tensor)
    
model =SimpleModel
params = model.init(jax.random.PRNGKey(0))
tensor_train = params['params']['tensor']
print("Trianble parameter",tensor_train)