# JAX-MNIST Neural Network Training

This is a simple implementation of a convolutional neural network using JAX and Flax to train on the MNIST dataset.

## Requirements:

- **JAX**: A high-performance machine learning library. [Documentation](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- **Flax**: A neural network library for JAX.
- **optax**: A library for optimization routines, used in conjunction with JAX.
- **tensorflow_datasets**: Library for easily accessing and manipulating datasets.

## Installation:

To install the required dependencies, run the following:

```
pip install jax jaxlib flax optax tensorflow tensorflow_datasets
```

## Components:

1. **Net Class (Model Definition)**:

   - A simple convolutional neural network with the following layers:
     - 2 Convolution layers.
     - 2 Dense layers.
     - Dropout layer.
   - The `__call__` function defines the forward pass of the network.

2. **main Function**:
   - Responsible for the entire pipeline, including:
     - Data loading and normalization using `tensorflow_datasets`.
     - Model initialization.
     - Training loop for 5 epochs.
     - Evaluation on the test set.

## Usage:

To execute the script, simply run:

```bash
python mnist_jax.py
```

## Outputs:

- The script will print the training loss every 100 batches.
- At the end of the training, it will print the test accuracy.

```
Epoch 5, Loss: 0.0932893231511116
Epoch 5, Loss: 0.10563230514526367
Test accuracy: 98.74%
```

## Notes:

1. Model uses the Adam optimizer with a learning rate of `0.001`.
2. Dropout is applied during training but not during evaluation.
3. The script uses `optax` for optimization and the `softmax_cross_entropy` loss function for training.
4. For reproducibility, the random seed (PRNGKey) is set to `0`.

## Future Improvements:

1. Integrate validation data to monitor overfitting during training.
2. Implement model saving and loading functionalities.
3. Add more complex architectures or regularization techniques for improved performance.
4. Allow hyperparameters (like learning rate, batch size, etc.) to be set via command line arguments.
