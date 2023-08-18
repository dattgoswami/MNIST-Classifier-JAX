import jax
import jax.numpy as jnp
from jax import random, jit, grad
import flax.linen as nn
import optax
import tensorflow_datasets as tfds

class Net(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.fc1 = nn.Dense(features=128)
        self.fc2 = nn.Dense(features=10)
        self.dropout = nn.Dropout(rate=0.5)

    def __call__(self, x, training=True, dropout_key=None):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not training, rng=dropout_key)
        x = self.fc2(x)
        return x

def main():
    key = random.PRNGKey(0)

    # Load the MNIST dataset
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

    # Normalize data
    train_images, train_labels = train_ds['image'], train_ds['label']
    test_images, test_labels = test_ds['image'], test_ds['label']

    train_images = jnp.float32(train_images) / 255.0
    test_images = jnp.float32(test_images) / 255.0

    # Initialize model and optimizer
    model = Net()
    rngs = {'params': key, 'dropout': key}
    params = model.init(
        rngs, train_images[0:1])
    tx = optax.adam(0.001)
    opt_state = tx.init(params)

    num_epochs = 5
    batch_size = 32

    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(train_images), batch_size):
            batch = {
                'image': train_images[i:i+batch_size],
                'label': train_labels[i:i+batch_size],
            }

            # Split key for dropout
            key, dropout_key = random.split(key)

            # Compute loss
            logits = model.apply(params, batch['image'],
                     training=True, dropout_key=dropout_key)

            one_hot = jax.nn.one_hot(batch['label'], 10)
            loss = jnp.mean(optax.softmax_cross_entropy(
                logits=logits, labels=one_hot))

            # Compute gradients
            grads = grad(lambda p, i, l, k: jnp.mean(optax.softmax_cross_entropy(logits=model.apply(
                p, i, training=True, dropout_key=k), labels=jax.nn.one_hot(l, 10))))(params, batch['image'], batch['label'], dropout_key)

            # Update parameters
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if i % (batch_size * 100) == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

    # Evaluation
    test_logits = model.apply(params, test_images, training=False)
    test_predictions = jnp.argmax(test_logits, axis=-1)
    accuracy = jnp.mean(test_predictions == test_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
