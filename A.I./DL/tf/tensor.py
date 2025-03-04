#Tensorflow
import numpy as np 
import tensorflow as tf

###Immutable tensor (fixed values, not trainable).tf.constant()
# Rank 0 tensor
rank_0_tensor = tf.constant(2)
# Rank 1 tensor
rank_1_tensor = tf.constant([1,2,3])
# Rank 2 tensor 
rank_2_tensor = tf.constant([[1,2,3],
                             [4,5,6]])
# Rank 3 tensor
rank_3_tensor = tf.constant([
                            [[1,2,3],
                            [4,5,6.0]],

                            [[10,20,30],
                            [40,50,60.0]],

                            [[11,22,33],
                            [44,55,66.0]],
                            ])
# Tensorflow to Numpy array 
rank_2_tensor = tf.constant([[1,2,3],
                             [4,5,6.0]])
tf_to_numpy_array = np.array(rank_2_tensor)

#Basic math operation 
a = tf.constant([[11,12,13],
                 [14,15,16]])
b = tf.constant([[10,24,32],
                 [41,15,16]])

# 2x3
x = tf.constant([[11, 12, 13],
                 [14, 15, 16]])

#3x2 
y = tf.constant([[10, 24],
                 [41, 15],
                 [32, 16]])
sum_a_b = tf.add(a,b)
subtract_a_b = tf.subtract(a,b)
multiply_a_b = tf.multiply(a,b)
divide_a_b = tf.divide(a,b)
tensor_product = tf.matmul(x,y)


### Operation Min, Max
tensor_x = tf.constant([[100, 12.0, 13],
                 [14.0, 15, 160.0]])

tensor_y = tf.constant([[1.0, 2.0, 3.0],
                 [14.0, 5.0, 6.0]])

#Get index for the Max
tensor_max_indice = tf.argmax(tensor_x, axis=0)
#Get index for the Max value axis 0
tensor_max_axis = tf.reduce_max(tensor_x, axis=0)
#Get Min by axis
tensor_max = tf.reduce_min(tensor_x)
#Get Softmax
tensor_nn = tf.nn.softmax(tensor_y)
tensor_rank_4 = tf.constant([1,2,2,224])

###Tensorflow Shapes 

# Image: Batch of 8 RGB images (224x224)
image_tensor = tf.random.normal([8, 224, 224, 3])  # (N, H, W, C)

# Text: Batch of 16 sentences, each tokenized to 128 tokens
text_tensor = tf.random.uniform([16, 128], maxval=30522, dtype=tf.int32)  # (N, S)

# Audio: Batch of 4 audio clips, 16000 time steps, 80 features
audio_tensor = tf.random.normal([4, 16000, 80])  # (N, T, F)

# Video: Batch of 2 videos, 16 frames, RGB (224x224)
video_tensor = tf.random.normal([2, 16, 224, 224, 3])  # (N, T, H, W, C)

""""
| **Task**                 | **Tensor Shape (TensorFlow)** | **Batch, Height, Width, Channels**                 |
|--------------------------|-------------------------------|-----------------------------------------------------|
| **Image (CNNs, Vision)** | `(N, H, W, C)`                | `Batch = N`, `Height = H`, `Width = W`, `Channels = C` |
| **Text (Tokenized NLP)** | `(N, S)`                      | `Batch = N`, `Height = 1`, `Width = 1`, `Channels = S` |
| **Audio (Speech)**       | `(N, T, F)`                   | `Batch = N`, `Height = 1`, `Width = T`, `Channels = F` |
| **Video (Frames)**       | `(N, T, H, W, C)`             | `Batch = N`, `Height = H`, `Width = W`, `Channels = C` |

"""

### Slice tensor 
# Slice the first row
slice_1 = y[0]  # Output: tf.Tensor([10 24], shape=(2,), dtype=int32)

# Slice the first column
slice_2 = y[:, 0]  # Output: tf.Tensor([10 41 32], shape=(3,), dtype=int32)

# Slice the second row
slice_3 = y[1, :]  # Output: tf.Tensor([41 15], shape=(2,), dtype=int32)

# Slice a submatrix (first two rows)
slice_4 = y[0:2, :]  # Output: tf.Tensor([[10 24] [41 15]], shape=(2, 2), dtype=int32)

#Mutable tensor for trainable parameters
y_var = tf.Variable(y)
#Tag Variable
x_var_ = tf.Variable(x, name='Inmutable' )
## Trainable Weights
w = tf.Variable([[1.], [2.]])
z = tf.constant([[3., 4.]])
t = tf.Variable(z * 10, name='Multiplied')

"""
machine learning model 
    trainable model parameters
    training steps

"""

with tf.device('CPU:0'):

    r = tf.Variable([[1,2],[3,4]])
    o = tf.constant([[1,2],[3,4]])

    g = tf.matmul(r,o)


### NN
#Computing Gradients
#Define Scalar 
scalar = tf.Variable(2.5)
# Record Gradient 
with tf.GradientTape() as tape:
    y = scalar**2 
#Gradient of y with respect to the scalar 
dy_dx = tape.gradient(y,scalar)
print(dy_dx.numpy())


# weights
w = tf.Variable(tf.random.normal((3,2)), name = 'w')
print("w:", w)
# bias 
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
print("b:", b)

x = tf.constant([[1.0,2.0,3.0]])

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)


##Gradients
[dl_dw, dl_db] = tape.gradients(loss, [w,b])

#Diccionary 
dic_var = {
    'w': w,
    'b': b
}

gad_two = tf.gradient(loss, dic_var)
print("dl_dw", grad['w'])
print("dl_db", grad['b'])


if __name__ == "__main__":
   
  
    print("******INMUTABLE TENSORS********")
    print(rank_0_tensor)
    print(rank_1_tensor) 
    print(rank_2_tensor)
    print("///////////////")
    print(rank_3_tensor)
    print("Converted tensorflow to numpy")
    print(tf_to_numpy_array)
    print("SUM/ADD")
    print(sum_a_b)
    print("Substraction")
    print(subtract_a_b)
    print("Multiply")
    print(multiply_a_b)
    print("divide")
    print(divide_a_b)
    print("Tensor Product")
    print(tensor_product)
    print(tensor_nn)
    print("*****Slice.numpy()******")
    print("Slice 1:", slice_1.numpy())
    print("Slice 2:", slice_2.numpy())
    print("Slice 3:", slice_3.numpy())
    print("Slice 4:", slice_4.numpy())
    print("*****Slice******")
    print("Slice 1:", slice_1)
    print("Slice 2:", slice_2)
    print("Slice 3:", slice_3)
    print("Slice 4:", slice_4)
    print("#######MUTABLE TENSORS#######")
    print(y_var)
    print("*******")
    print(f"Tensor Z :{z},Tensor:{t},Tensor:{w}")
    """
    
    """
