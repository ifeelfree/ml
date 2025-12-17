
import numpy as np


import matplotlib
# Try one of these:
matplotlib.use('TkAgg')    # Tkinter backend
import matplotlib.pyplot as plt


def calibrate_images_least_squares(image_a, image_b):
    """
    Calibrates image_a to match image_b using a direct least-squares solution.
    Model: A_calibrated = A * alpha + beta.

    This function finds the optimal alpha and beta for each color channel (R, G, B)
    analytically.

    Args:
        image_a (np.ndarray): The source RGB image to be calibrated (H, W, 3).
        image_b (np.ndarray): The target RGB image (H, W, 3).

    Returns:
        tuple: A tuple containing:
            - calibrated_image (np.ndarray): The calibrated version of image_a as a uint8 NumPy array.
            - alphas (np.ndarray): The calculated alpha values for (R, G, B).
            - betas (np.ndarray): The calculated beta values for (R, G, B).
    """
    print("Solving for alpha and beta using least squares...")

    # Ensure calculations are done in floating point to avoid errors
    image_a_float = image_a.astype(np.float64)
    image_b_float = image_b.astype(np.float64)

    alphas = []
    betas = []

    # Solve for each channel independently
    for channel in range(3):
        # 1. Flatten the image data for the current channel
        a_flat = image_a_float[:, :, channel].flatten()
        b_flat = image_b_float[:, :, channel].flatten()

        # 2. Construct the X matrix
        # The first column is the source pixel values (for alpha).
        # The second column is all ones (for beta).
        X = np.stack([a_flat, np.ones_like(a_flat)], axis=1)

        # 3. Solve the least-squares problem: X * p = b_flat
        # np.linalg.lstsq returns the parameter vector p that minimizes the error.
        p, _, _, _ = np.linalg.lstsq(X, b_flat, rcond=None)

        alpha_channel, beta_channel = p
        alphas.append(alpha_channel)
        betas.append(beta_channel)

    alphas = np.array(alphas)
    betas = np.array(betas)

    print("Solving finished.")

    # 4. Apply the calculated transformation
    # Use broadcasting to apply the per-channel alpha and beta
    calibrated_image_float = image_a_float * alphas + betas

    # Clip the values to the valid [0, 255] range and convert back to uint8
    calibrated_image = np.clip(calibrated_image_float, 0, 255).astype(np.uint8)

    return calibrated_image, alphas, betas



if __name__ == '__main__':
    # --- Create a demonstration scenario ---
    # Let's create a sample image A (e.g., a gradient)
    height, width = 256, 256
    base_gradient = np.linspace(0, 255, width, dtype=np.uint8)
    image_a = np.tile(base_gradient, (height, 1))
    image_a = np.stack([image_a, image_a, image_a], axis=-1)  # Make it 3-channel

    # Now, let's create a target image B by applying a known transformation to A
    # This way, we can check if our function learns the correct parameters.
    alpha_true = np.array([0.8, 0.95, 1.1])  # Different scaling for each channel
    beta_true_in_0_255_range = np.array([20, -10, 5])  # Different offset for each channel

    # Apply the transformation in floating point space
    image_b = image_a.astype(np.float32) * alpha_true + beta_true_in_0_255_range
    # Add a small amount of noise to make the problem more realistic
    image_b += np.random.normal(0, 2, image_b.shape)
    # Clip and convert back to a valid image format
    image_b = np.clip(image_b, 0, 255).astype(np.uint8)

    # --- Run the calibration ---

    calibrated_a, alpha_learned, beta_learned = calibrate_images_least_squares(image_a,
                                                                                 image_b)

    # --- Print and display the results ---
    print("\n--- Results ---")
    print(f"True Alpha:      {alpha_true}")
    print(f"Learned Alpha:   {alpha_learned}")
    print(f"\nTrue Beta (in 0-255 range): {beta_true_in_0_255_range}")
    print(f"Learned Beta (in 0-255 range):  {beta_learned}")

    # Visualize the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_a)
    axes[0].set_title("Original Image A")
    axes[0].axis('off')

    axes[1].imshow(image_b)
    axes[1].set_title("Target Image B")
    axes[1].axis('off')

    axes[2].imshow(calibrated_a)
    axes[2].set_title("Calibrated Image A")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
