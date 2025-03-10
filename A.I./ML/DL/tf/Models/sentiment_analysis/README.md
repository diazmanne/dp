# Standard Operating Procedure (S.O.P) for Binary Classification ML System

## 1. Introduction

This document provides a step-by-step guide to setting up and running a binary classification machine learning model using TensorFlow and Jupyter Notebook.

## 2. Prerequisites

Ensure you have the following installed on your system:

- Python (>=3.8)
- TensorFlow (>=2.x)
- Pandas, NumPy
- Jupyter Notebook
- Kaggle API (to download dataset)

To install the required packages, run:

```bash
pip install tensorflow pandas numpy jupyter
```

## 3. File Structure

Ensure the script is located at:

```
~/Workspace/dp/A.I./ML/DL/tf/Models/sentiment_analysis/
```

## 4. Pseudocode Overview

```
1. Load and explore dataset
2. Preprocess dataset (clean, tokenize, vectorize)
3. Split dataset into training, validation, and test sets
4. Define model architecture using TensorFlow
5. Compile and train the model
6. Evaluate the model
7. Generate predictions and analyze results
```

## 5. Step-by-Step: Building and Testing in Jupyter Notebook

### Step 1: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 2: Load Dataset

```python
import pandas as pd

data_df = pd.read_csv("FinalBalancedDataset.csv")
data_df = data_df.drop(columns=["Unnamed: 0"])
print(data_df.head())
```

### Step 3: Preprocess Data

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_df["tweet"])
tokenized_output = tokenizer.texts_to_sequences(data_df["tweet"])
padded_sequences = pad_sequences(tokenized_output, maxlen=484)

# Labels
tags = data_df["Toxicity"].values
```

### Step 4: Convert to TensorFlow Dataset

```python
dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, tags))
dataset = dataset.cache().shuffle(len(padded_sequences)).batch(16)

train = dataset.take(int(len(dataset) * 0.7))
val = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.3))
test = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))
```

### Step 5: Define Model Architecture

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense

input_ids = Input(shape=(484,), dtype=tf.int32, name="input_ids")

x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32)(input_ids)
x = Bidirectional(LSTM(32, activation='tanh'))(x)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_ids, outputs=output, name="text-classifier")
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
model.summary()
```

### Step 6: Train Model

```python
def map_dataset_to_input_and_labels(dataset):
    return dataset.map(lambda x, y: ({'input_ids': x}, y))

train = map_dataset_to_input_and_labels(train)
val = map_dataset_to_input_and_labels(val)

history = model.fit(
    train,
    epochs=1,
    batch_size=16,
    validation_data=val
)
```

### Step 7: Evaluate Model

```python
results = model.evaluate(test)
print("Test Loss, Test Accuracy:", results)
```

### Step 8: Generate Predictions

```python
sample_text = "This is a test sentence."
tokenized_sample = tokenizer.texts_to_sequences([sample_text])
padded_sample = pad_sequences(tokenized_sample, maxlen=484)
prediction = model.predict({"input_ids": padded_sample})
print("Predicted Toxicity Score:", prediction)
```

## 6. Model Results Explanation

### Model Summary

```
Model: "text-classifier"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_ids (InputLayer)      [(None, 484)]                0         []                            
                                                                                                  
 embedding_1 (Embedding)     (None, 484, 32)              927872    ['input_ids[0][0]']           
                                                                                                  
 bidirectional_1 (Bidirecti  (None, 64)                   16640     ['embedding_1[0][0]']         
 onal)                                                                                           
                                                                                                  
 dense_4 (Dense)             (None, 128)                  8320      ['bidirectional_1[0][0]']     
                                                                                                  
 dense_5 (Dense)             (None, 256)                  33024     ['dense_4[0][0]']             
                                                                                                  
 dense_6 (Dense)             (None, 128)                  32896     ['dense_5[0][0]']             
                                                                                                  
 dense_7 (Dense)             (None, 1)                    129       ['dense_6[0][0]']             
                                                                                                  
==================================================================================================
Total params: 1018881 (3.89 MB)
Trainable params: 1018881 (3.89 MB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
```

### Training Performance

```
2482/2482 [==============================] - 1290s 518ms/step - loss: 0.2020 - accuracy: 0.9197 - val_loss: 0.1238 - val_accuracy: 0.9560
```

- The final model achieved a **training accuracy of 91.97%** and a **validation accuracy of 95.60%**.
- The loss function indicates how well the model is performing, with lower values being better.
- The model successfully learns patterns from input sequences and generalizes well to unseen data.

## 7. Conclusion

This S.O.P guides users through setting up and running a binary classification ML model in Jupyter Notebook. Once successfully replicated, additional automation and pipeline integration will be considered.


