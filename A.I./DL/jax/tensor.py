#JAX
import jax 
import numpy as np 
import jax.numpy as jnp


# Rank 0 tensor
rank_0_tensor = jnp.array(2)
rank_1_tensor = jnp.array([1,2,3])
# Rank 2 tensor 
rank_2_tensor = jnp.array([[1,2,3],
                             [4,5,6.0]])
# Rank 3 tensor
rank_3_tensor = jnp.array([
                            [[1,2,3],
                            [4,5,6.0]],

                            [[10,20,30],
                            [40,50,60.0]],

                            [[11,22,33],
                            [44,55,66.0]],
                            ])

# Tensorflow to Numpy array 
tf_to_numpy_array = np.array(rank_2_tensor)

#Basic math operation 
a = jnp.array([[11,12,13],
                 [14,15,16]])
b = jnp.array([[10,24,32],
                 [41,15,16]])
# 2x3
x = jnp.array([[11, 12, 13],
                 [14, 15, 16]])
# 3x2 
y = jnp.array([[10, 24],
                 [41, 15],
                 [32, 16]])
# Sum of a and b
sum_a_b = jnp.add(a, b)

# Subtraction of a and b
subtract_a_b = jnp.subtract(a, b)

# Element-wise multiplication of a and b
multiply_a_b = jnp.multiply(a, b)

# Element-wise division of a and b
divide_a_b = jnp.divide(a, b)

# Matrix multiplication of x and y
tensor_product = jnp.matmul(x, y)
tensor_rank_4 = jnp.array([1,224,224,3])
### Tensor Shapes 
# Image: Batch of 8 RGB images (224x224)
image_tensor = jax.random.normal(jax.random.PRNGKey(0), (8, 224, 224, 3))  # (N, H, W, C)

# Text: Batch of 16 sentences, each tokenized to 128 tokens
text_tensor = jax.random.randint(jax.random.PRNGKey(1), (16, 128), 0, 30522)  # (N, S)

# Audio: Batch of 4 audio clips, 16000 time steps, 80 features
audio_tensor = jax.random.normal(jax.random.PRNGKey(2), (4, 16000, 80))  # (N, T, F)

# Video: Batch of 2 videos, 16 frames, RGB (224x224)
video_tensor = jax.random.normal(jax.random.PRNGKey(3), (2, 16, 224, 224, 3))  # (N, T, H, W, C)

"""
| **Task**                 | **Tensor Shape (JAX)**      | **Batch, Height, Width, Channels**                 |
|--------------------------|-----------------------------|-----------------------------------------------------|
| **Image (CNNs, Vision)** | `(N, H, W, C)`              | `Batch = N`, `Height = H`, `Width = W`, `Channels = C` |
| **Text (Tokenized NLP)** | `(N, S)`                    | `Batch = N`, `Height = 1`, `Width = 1`, `Channels = S` |
| **Audio (Speech)**       | `(N, T, F)`                 | `Batch = N`, `Height = 1`, `Width = T`, `Channels = F` |
| **Video (Frames)**       | `(N, T, H, W, C)`           | `Batch = N`, `Frames = T`, `Height = H`, `Width = W`, `Channels = C` |

"""
 #Slice the first row
slice_1 = y[0]  # First row

# Slice the first column
slice_2 = y[:, 0]  # First column

# Slice the second row
slice_3 = y[1, :]  # Second row

# Slice a submatrix (first two rows)
slice_4 = y[0:2, :]  # First two rows











if __name__ == "__main__":
    
    
    print(f"Image Tensor: {image_tensor}, Shape:{image_tensor.shape}, Dtype:{image_tensor.dtype} ")
    print("***************************************")
    print("***************************************")
    # f-strings f"String:{var.method, String:var.method, ...}"
    print(f"Tensor 0:{rank_0_tensor}, Tensor 1:{rank_1_tensor}, Shape:{rank_1_tensor.shape}, Dtype:{rank_1_tensor.dtype}")
    #f"{String:rank_0_tensor.shape}""
    print(f"JAX Tensor: {rank_0_tensor}, Shape: {rank_0_tensor.shape}, Dtype: {rank_0_tensor.dtype}")
    print(f"Tensor 2:{rank_2_tensor}, Shape :{rank_2_tensor.shape}, Dtype:{rank_2_tensor.dtype}")
    print(f"Tensor 3:{rank_3_tensor}, Shape :{rank_3_tensor.shape}, Dtype:{rank_3_tensor.dtype}")
    print("Converted tensorflow to numpy")
    print(f"Tensor 2:{tf_to_numpy_array}, Shaped: {tf_to_numpy_array.shape}, Type:{tf_to_numpy_array.dtype}")
    print("Sum:\n", sum_a_b)
    print("Subtraction:\n", subtract_a_b)
    print("Multiplication:\n", multiply_a_b)
    print("Division:\n", divide_a_b)
    print("Matrix Product:\n", tensor_product)
    print("*****Slice ***")
    print(f"Slice 1: {slice_1}, Slice 2: {slice_2}, Slice 3: {slice_3}, Slice 4: {slice_4}")
    """
    2
    JAX Tensor: 2, Shape: (), Dtype: int32
    """