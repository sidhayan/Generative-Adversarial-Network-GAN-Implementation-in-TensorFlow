import tensorflow as tf
from tensorflow.keras import layers

# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_shape=(100,), activation='relu'))
    model.add(layers.Dense(784, activation='sigmoid'))  # Output layer, representing generated image
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, input_shape=(784,), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer, classifying real or fake
    return model

# Creating instances of generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Setting up the GAN
noise = tf.random.normal([1, 100])  # Generating random noise as input to the generator
generated_image = generator(noise)  # Generating an image using the generator

decision = discriminator(generated_image)  # Discriminator's decision on the generated image

print(f"Generated Image Shape: {generated_image.shape}")
print(f"Discriminator Decision: {decision.numpy()}")
