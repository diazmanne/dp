DEEP LEARNING 
0.Subset of machine learning using multi-layered neural networks

Model, loss function/Objective function , and optimization algorithm

.
├── Deep Learning Taxonomy (Ordered by Complexity)
│   ├── Supervised Learning
│   │   ├── Feedforward Neural Networks (FNN)  # Basic & Fully Connected
│   │   │   ├── *Components:* Fully connected layers, Activation functions (ReLU, Sigmoid), Loss function (MSE, Cross-Entropy)
│   │   │   ├── 'Optimization:' Stochastic Gradient Descent (SGD), Adam, RMSprop
│   │   ├── Convolutional Neural Networks (CNN)  # Adds spatial awareness
│   │   │   ├── *Components:* Convolutional layers, Pooling layers, Fully connected layers, Activation functions (ReLU, Softmax), Loss function
│   │   │   ├── 'Optimization:' Adam, SGD with momentum, Learning rate scheduling
│   │   ├── Recurrent Neural Networks (RNN)  # Introduces sequential processing
│   │   │   ├── *Components:* Recurrent layers, Memory state, Activation functions (Tanh, ReLU), Loss function (Cross-Entropy, MSE)
│   │   │   ├── 'Optimization:' RMSprop, Adam, Gradient clipping to prevent exploding gradients
│   │   │   ├── Long Short-Term Memory (LSTM)  # Advanced memory control
│   │   │   │   ├── *Components:* LSTM cells, Input/output gates, Forget gate, Loss function
│   │   │   │   ├── 'Optimization:' Adam, RMSprop, Layer normalization
│   │   │   └── Gated Recurrent Units (GRU)  # Simplified LSTM
│   │   │       ├── *Components:* GRU cells, Update/reset gates, Activation functions, Loss function
│   │   │       ├── 'Optimization:' Adam, Adaptive learning rates, Clipping gradients
│   │   ├── Transformers  # Scalable attention-based learning
│   │   │   ├── *Components:* Self-attention layers, Positional encoding, Multi-head attention, Feedforward layers, Loss function
│   │   │   ├── 'Optimization:' AdamW, Learning rate warm-up, Cosine decay
│   │   └── Graph Neural Networks (GNN)  # Works with graph-structured data
│   │       ├── *Components:* Graph convolutions, Node/edge embeddings, Aggregation functions, Loss function
│   │       ├── 'Optimization:' Adam, Gradient-based optimization with specialized graph sparsity tricks
│   ├── Unsupervised Learning
│   │   ├── Autoencoders  # Basic feature extraction
│   │   │   ├── *Components:* Encoder-decoder architecture, Bottleneck layer, Loss function (Reconstruction loss)
│   │   │   ├── 'Optimization:' Adam, Mean Squared Error minimization
│   │   │   └── Variational Autoencoders (VAE)  # Adds probabilistic learning
│   │   │       ├── *Components:* Probabilistic encoder/decoder, Latent variable sampling, KL divergence loss
│   │   │       ├── 'Optimization:' Variational Inference, Adam, Stochastic Variational Bayes
│   │   ├── Self-Organizing Maps (SOM)  # Competitive learning
│   │   │   ├── *Components:* Neuron grid, Competitive learning, Weight updates (Winner-Takes-All)
│   │   │   ├── 'Optimization:' Kohonen learning rule, Decaying learning rate
│   │   ├── Deep Belief Networks (DBN)  # Stacked RBMs
│   │   │   ├── *Components:* Stacked Restricted Boltzmann Machines (RBM), Probabilistic layer-wise training, Loss function
│   │   │   ├── 'Optimization:' Contrastive Divergence (CD), Persistent CD (PCD), Adam
│   │   ├── Generative Adversarial Networks (GAN)  # Adversarial training
│   │   │   ├── *Components:* Generator, Discriminator, Adversarial loss (Binary Cross-Entropy)
│   │   │   ├── 'Optimization:' Minimax optimization, Gradient Penalty, Adam, RMSprop
│   ├── Reinforcement Learning
│   │   ├── Deep Q Networks (DQN)  # Basic RL with deep learning
│   │   │   ├── *Components:* Q-value approximation, Experience replay, Target network, Loss function
│   │   │   ├── 'Optimization:' Q-learning, Adam, Bellman equation updates
│   │   ├── Policy Gradient Methods  # Direct policy optimization
│   │   │   ├── *Components:* Policy network, Reward function, Gradient-based updates, Loss function
│   │   │   ├── 'Optimization:' Monte Carlo policy gradient, Adam, Advantage normalization
│   │   │   ├── Actor-Critic  # Adds value-based learning
│   │   │   │   ├── *Components:* Actor network, Critic network, Advantage function, Loss function
│   │   │   │   ├── 'Optimization:' Actor-critic loss function, Adam, PPO (Proximal Policy Optimization)
│   │   │   └── Asynchronous Advantage Actor-Critic (A3C)  # Multi-agent learning
│   │   │       ├── *Components:* Multiple asynchronous agents, Shared network, Loss function
│   │   │       ├── 'Optimization:' Asynchronous parallel training, RMSprop, Adaptive learning rate
│   │   └── Deep Deterministic Policy Gradient (DDPG)  # Continuous action space
│   │       ├── *Components:* Actor-critic architecture, Off-policy learning, Replay buffer, Loss function
│   │       ├── 'Optimization:' Twin Delayed DDPG (TD3), Adam, Target network updates
│   └── Hybrid & Emerging Models (Most Complex)
│       ├── Neural Turing Machines (NTM)  # External memory learning
│       │   ├── *Components:* Memory matrix, Controller (RNN), Read/Write heads, Loss function
│       │   ├── 'Optimization:' Reinforcement learning + SGD, Gradient-based memory updates
│       ├── Memory Networks  # Long-term reasoning
│       │   ├── *Components:* Memory modules, Attention mechanisms, Query-response structure, Loss function
│       │   ├── 'Optimization:' End-to-End Memory Network optimization, Adam, SGD
│       └── Capsule Networks  # Hierarchical spatial understanding
│           ├── *Components:* Capsules, Dynamic routing, Squashing function, Loss function
│           ├── 'Optimization:' Routing-by-agreement, Adam, Specialized loss functions like Margin Loss
