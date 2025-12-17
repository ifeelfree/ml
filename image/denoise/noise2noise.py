import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib
matplotlib.use('TkAgg')    # Tkinter backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

if __name__ == "__main__":


    # --- 1. Configuration and Data Loading ---

    # Parameters
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 128
    NOISE_LEVEL = 0.4  # Standard deviation of the Gaussian noise
    EPOCHS = 10

    # Load the MNIST dataset (we only need the images, not the labels)
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train[0:10,...]
    x_test = x_test[0:10,...]


    # Function to normalize and reshape data
    def preprocess_images(images):
        images = images.astype("float32") / 255.0
        images = np.reshape(images, (len(images), 28, 28, 1))
        return images


    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")


    # --- 2. Noise Generation and Dataset Pipeline ---

    # Function to add noise to a batch of images
    def add_noise(images):
        noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=NOISE_LEVEL, dtype=tf.float32)
        noisy_images = images + noise
        # Clip the results to be between 0 and 1
        noisy_images = tf.clip_by_value(noisy_images, 0.0, 1.0)
        return noisy_images


    # Function to create the Noise2Noise training pairs
    def create_noisy_pairs(clean_image):
        # For each clean image, generate two different noisy versions
        noisy_input = add_noise(clean_image)
        noisy_target = add_noise(clean_image)
        return noisy_input, noisy_target


    # Create the training dataset
    # We start with the clean images and use .map() to generate noisy pairs on the fly.
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.map(create_noisy_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    # Create the validation dataset
    # For validation, we want to compare the model's output to the original clean image
    # to see how well it's actually denoising.
    # Input: noisy image, Target: clean image
    def create_noisy_clean_pairs(clean_image):
        noisy_input = add_noise(clean_image)
        return noisy_input, clean_image


    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.map(create_noisy_clean_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    # --- 3. Build the Denoising Model (Simple U-Net) ---

    def build_denoising_model():
        inputs = layers.Input(shape=IMG_SHAPE)

        # Encoder
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)

        # Bottleneck
        bottleneck = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

        # Decoder
        up1 = layers.UpSampling2D((2, 2))(bottleneck)
        concat1 = layers.Concatenate()([up1, conv2])  # Skip connection
        conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)

        up2 = layers.UpSampling2D((2, 2))(conv3)
        concat2 = layers.Concatenate()([up2, conv1])  # Skip connection
        conv4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)

        # Output layer
        outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv4)

        model = keras.Model(inputs, outputs)
        return model


    model = build_denoising_model()
    model.summary()

    # --- 4. Compile and Train the Model ---

    # We use Mean Squared Error as the loss function, as proven in the paper's theory.
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset
    )
    print("Training finished.")


    # --- 5. Perform Inference and Visualize Results ---

    def display_results(images_to_show, num_images=5):
        # Get a batch of test images
        original_clean = images_to_show[:num_images]

        # Create noisy versions for input
        noisy_input = add_noise(original_clean)

        # Get the model's prediction
        denoised_output = model.predict(noisy_input)

        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            # Display original clean image
            ax = plt.subplot(3, num_images, i + 1)
            plt.imshow(original_clean[i].reshape(28, 28), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_title("Original Clean", fontdict={'fontsize': 12})

            # Display noisy input image
            ax = plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(noisy_input[i].numpy().reshape(28, 28), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_title("Noisy Input", fontdict={'fontsize': 12})

            # Display denoised output
            ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
            plt.imshow(denoised_output[i].reshape(28, 28), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_title("Denoised Output", fontdict={'fontsize': 12})

        plt.tight_layout()
        plt.show()


    print("\nDisplaying inference results...")
    display_results(x_test)
