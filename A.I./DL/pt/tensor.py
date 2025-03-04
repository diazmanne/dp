#Pytorch
import numpy as np 
import torch

###Immutable tensor (fixed values, not trainable).torch.tensor
# Rank 0 tensor
rank_0_tensor = torch.tensor(2)
# Rank 1 tensor
rank_1_tensor = torch.tensor([1,2,3])
# Rank 2 tensor 
rank_2_tensor = torch.tensor([[1,2,3],
                             [4,5,6.0]])
# Rank 3 tensor
rank_3_tensor = torch.tensor([
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
a = torch.tensor([[11,12,13],
                 [14,15,16]])
b = torch.tensor([[10,24,32],
                 [41,15,16]])
# 2x3
x = torch.tensor([[11, 12, 13],
                 [14, 15, 16]])
# 3x2 
y = torch.tensor([[10, 24],
                 [41, 15],
                 [32, 16]])
y_float = torch.tensor([[1.0, 2.4],
                 [4.1, 1.5],
                 [3.2, 1.6]])
tensor_float = torch.randn(8,2,2,3,dtype=torch.float32)

sum_a_b = torch.add(a,b)
subtract_a_b = torch.sub(a,b)
multiply_a_b = torch.mul(a,b)
divide_a_b = torch.div(a,b)
tensor_product = torch.matmul(x,y)
tensor_rank_4 = torch.zeros([1,3,224,224])

### Tensor Shapes 
# Image: Batch of 8 RGB images (224x224)
image_tensor = torch.randn(8, 3, 224, 224)  # (N, C, H, W)
# Text: Batch of 16 sentences, each tokenized to 128 tokens
text_tensor = torch.randint(0, 30522, (16, 128))  # (N, S)
# Audio: Batch of 4 audio clips, 16000 time steps, 80 features
audio_tensor = torch.randn(4, 80, 16000)  # (N, F, T)
# Video: Batch of 2 videos, 16 frames, RGB (224x224)
video_tensor = torch.randn(2, 16, 3, 224, 224)  # (N, T, C, H, W)

"""
| **Task**                 | **Tensor Shape (PyTorch)**  | **Batch, Height, Width, Channels**                 |
|--------------------------|-----------------------------|-----------------------------------------------------|
| **Image (CNNs, Vision)** | `(N, C, H, W)`              | `Batch = N`, `Channels = C`, `Height = H`, `Width = W` |
| **Text (Tokenized NLP)** | `(N, S)`                    | `Batch = N`, `Height = 1`, `Width = 1`, `Channels = S` |
| **Audio (Speech)**       | `(N, F, T)`                 | `Batch = N`, `Height = 1`, `Width = T`, `Channels = F` |
| **Video (Frames)**       | `(N, T, C, H, W)`           | `Batch = N`, `Frames = T`, `Channels = C`, `Height = H`, `Width = W` |
"""

### Slice
# Slice the first row
slice_1 = y[0]  # Output: tensor([10, 24])
# Slice the first column
slice_2 = y[:, 0]  # Output: tensor([10, 41, 32])
# Slice the second row
slice_3 = y[1, :]  # Output: tensor([41, 15])
# Slice a submatrix (first two rows)
slice_4 = y[0:2, :]  # Output: tensor([[10, 24], [41, 15]])

###Mutable tensor (trainable variables).tf.Variable
y_var = torch.nn.Parameter(y_float)
##
w = torch.nn.Parameter(torch.tensor([[1.], [2.]]))
z = torch.tensor([[3., 4.]])
t = torch.nn.Parameter(z ** 2)


if __name__ == "__main__":
   
    print("########Inmutable Tensor############")
    print(rank_0_tensor)
    print(rank_0_tensor.shape)   # Output: torch.Size([])
    print(rank_0_tensor.dtype)   # Output: torch.int64 (default dtype in PyTorch)
    print("///////#Rank 1/////")
    print(rank_1_tensor.shape)   # Output: torch.Size([])
    print(rank_1_tensor.dtype)
    print(rank_1_tensor)
    print("///////#Rank 2/////")
    print(rank_2_tensor)
    print("///////#Rank 3/////")
    print(rank_3_tensor)
    print("Converted tensorflow to numpy")
    print(tf_to_numpy_array)
    print(tensor_float)
    print("*****************")
    print("Sum:\n", sum_a_b)
    print("Subtraction:\n", subtract_a_b)
    print("Multiplication:\n", multiply_a_b)
    print("Division:\n", divide_a_b)
    print("Matrix Product:\n", tensor_product)
    print("******SLICE********")
    print(slice_1)
    print(slice_2)
    print(slice_3)
    print(slice_4)
    print("******MUTABLE TENSORS********")
    print(y_var)
    print("*******")
    print("*******")
    print(f"Tensor Z :{z},Tensor:{t},Tensor:{w}")
    """
    tensor(2)
    tensor([1, 2, 3])
    torch.Size([])
    torch.int64
    ///////#Rank 1/////
    torch.Size([3])
    torch.int64

    """