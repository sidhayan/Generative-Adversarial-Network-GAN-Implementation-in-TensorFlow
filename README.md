# GAN Code Overview

## Generator Model
The generator model is a sequential Keras model with one hidden layer (128 neurons, ReLU activation) and an output layer (784 neurons, sigmoid activation) representing the generated image.

```python
# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_shape=(100,), activation='relu'))
    model.add(layers.Dense(784, activation='sigmoid'))  # Output layer, representing generated image
    return model
```
## Discriminator Model
The discriminator model is another sequential Keras model with one hidden layer (128 neurons, ReLU activation) and an output layer (1 neuron, sigmoid activation) classifying whether an input image is real or fake.

```python
# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_shape=(784,), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer, classifying real or fake
    return model
```

## Creating Instances and Setting up GAN
Instances of the generator and discriminator are created, and a random noise vector is used as input to the generator.
```python
# Creating instances of generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Setting up the GAN
noise = tf.random.normal([1, 100])  # Generating random noise as input to the generator
generated_image = generator(noise)  # Generating an image using the generator
decision = discriminator(generated_image)  # Discriminator's decision on the generated image
```
## Printing Results
The shape of the generated image and the discriminator's decision for that generated image are printed.
```python
# Printing Results
print(f"Generated Image Shape: {generated_image.shape}")
print(f"Discriminator Decision: {decision.numpy()}")

```
