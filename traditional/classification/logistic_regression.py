import numpy as np
import matplotlib
# Try one of these:
matplotlib.use('TkAgg')    # Tkinter backend

import matplotlib.pyplot as plt

def demo_sigmoid():


    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Visualize sigmoid
    z = np.linspace(-10, 10, 100)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(z, sigmoid(z))
    plt.grid(True, alpha=0.3)
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.title('Sigmoid Function')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Show sigmoid properties
    plt.subplot(1, 2, 2)
    special_points = [-5, -2, 0, 2, 5]
    plt.scatter(special_points, sigmoid(np.array(special_points)), color='red', s=100)
    for z_val in special_points:
        plt.annotate(f'σ({z_val})={sigmoid(z_val):.3f}',
                     xy=(z_val, sigmoid(z_val)),
                     xytext=(z_val + 0.5, sigmoid(z_val) + 0.05))
    plt.plot(z, sigmoid(z), alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('Key Sigmoid Values')
    plt.tight_layout()
    plt.show()

def demo_gradient_cal():

    # Small example: 3 samples, 2 features + bias
    X = np.array([[1, 2, 3],  # [bias, x1, x2] for sample 1
                  [1, 4, 5],  # [bias, x1, x2] for sample 2
                  [1, 6, 7]])  # [bias, x1, x2] for sample 3

    y = np.array([0, 1, 1])  # True labels
    y_pred = np.array([0.2, 0.8, 0.9])  # Predicted probabilities

    # Vectorized gradient computation
    gradient_vectorized = (1 / 3) * np.dot(X.T, (y_pred - y))
    print("Vectorized gradient:", gradient_vectorized)

    # Manual computation for verification
    gradient_manual = np.zeros(3)
    for j in range(3):  # For each weight
        sum_grad = 0
        for i in range(3):  # For each sample
            sum_grad += X[i, j] * (y_pred[i] - y[i])
        gradient_manual[j] = sum_grad / 3

    print("Manual gradient:    ", gradient_manual)
    print("Are they equal?", np.allclose(gradient_vectorized, gradient_manual))


import numpy as np


class LogisticRegression:
    """
    Logistic Regression with combined weight vector (bias included)
    Mathematical implementation matching our derivation above
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.losses = []

    def add_bias_term(self, X):
        """Add x₀=1 for bias term w₀"""
        n_samples = X.shape[0]
        return np.c_[np.ones(n_samples), X]

    def sigmoid(self, z):
        """σ(z) = 1 / (1 + e^(-z))"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y_true, y_pred):
        """J = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]"""
        epsilon = 1e-7  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Gradient Descent Optimization
        w := w - α·∇J(w)
        where ∇J(w) = (1/n)·X^T·(ŷ - y)
        """
        # Add bias term to X
        X = self.add_bias_term(X)
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)

        # Gradient descent
        for iteration in range(self.n_iterations):
            # Forward propagation: ŷ = σ(X·w)
            z = np.dot(X, self.weights)
            y_pred = self.sigmoid(z)

            # Compute loss
            loss = self.binary_cross_entropy(y, y_pred)
            self.losses.append(loss)

            # Backward propagation: ∇J = (1/n)·X^T·(ŷ - y)
            gradient = (1 / n_samples) * np.dot(X.T, (y_pred - y))

            # Update weights: w := w - α·∇J
            self.weights -= self.learning_rate * gradient

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """Calculate P(y=1|X)"""
        X = self.add_bias_term(X)
        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        return (self.predict_proba(X) >= threshold).astype(int)


# Generate synthetic dataset
def generate_dataset(n_samples=1000, n_features=5, random_state=42, show_fig=False):
    """Create a linearly separable binary classification dataset"""
    np.random.seed(random_state)

    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # Create labels based on a linear combination
    true_weights = np.array([0.5, 1.5, -2.0, 0.8, -1.2, 0.5])
    X_with_bias = np.c_[np.ones(n_samples), X]
    z = np.dot(X_with_bias, true_weights) + np.random.randn(n_samples) * 0.5
    y = (z > 0).astype(int)

    if show_fig:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Z values
        scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=z, cmap='coolwarm', alpha=0.6)
        axes[0].set_title('Z values (linear combination)')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        plt.colorbar(scatter1, ax=axes[0])

        # Plot 2: Decision boundary (z = 0)
        axes[1].scatter(X[:, 0], X[:, 1], c=z, cmap='coolwarm', alpha=0.6)
        # Draw decision boundary where z = 0
        x_boundary = np.linspace(-3, 3, 100)
        y_boundary = -(true_weights[0] + true_weights[1] * x_boundary) / true_weights[2]
        axes[1].plot(x_boundary, y_boundary, 'k--', linewidth=2, label='z = 0 (boundary)')
        axes[1].set_title('Decision Boundary')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].legend()
        axes[1].set_xlim(-3, 3)
        axes[1].set_ylim(-3, 3)

        # Plot 3: Binary labels
        axes[2].scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0 (z ≤ 0)', alpha=0.6)
        axes[2].scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1 (z > 0)', alpha=0.6)
        axes[2].plot(x_boundary, y_boundary, 'k--', linewidth=2, alpha=0.5)
        axes[2].set_title('Binary Classification')
        axes[2].set_xlabel('Feature 1')
        axes[2].set_ylabel('Feature 2')
        axes[2].legend()
        axes[2].set_xlim(-3, 3)
        axes[2].set_ylim(-3, 3)

        plt.tight_layout()
        plt.show()

        print(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")

    return X, y, true_weights


def plot_training_progress(model):
    """Visualize loss curve and decision boundary"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(model.losses)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Binary Cross-Entropy Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].grid(True, alpha=0.3)

    # Final weights
    weights = model.weights
    axes[1].bar(range(len(weights)), weights)
    axes[1].set_xlabel('Weight Index')
    axes[1].set_ylabel('Weight Value')
    axes[1].set_title('Learned Weights (w₀ is bias)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Main demonstration
def main():
    print("=" * 60)
    print("LOGISTIC REGRESSION: FROM MATH TO CODE")
    print("=" * 60)

    # Generate data
    X, y, true_weights = generate_dataset(n_samples=1000, show_fig=True)
    print(f"\n✓ Generated {len(X)} samples with {X.shape[1]} features")
    print(f"✓ Class distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Normalize features (important for gradient descent)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Train model
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)

    model = LogisticRegression(learning_rate=0.1, n_iterations=500, verbose=True)
    model.fit(X_train, y_train)

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Accuracy
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")

    # Detailed metrics for test set
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Show learned parameters
    print("\n" + "=" * 60)
    print("LEARNED PARAMETERS")
    print("=" * 60)
    print(f"Bias (w₀): {model.weights[0]:.4f}")
    print(f"Feature weights: {model.weights[1:]}")

    # Demonstrate probability predictions
    print("\n" + "=" * 60)
    print("PROBABILITY PREDICTIONS (5 samples)")
    print("=" * 60)

    for i in range(5):
        prob = model.predict_proba(X_test[i:i + 1])[0]
        pred = model.predict(X_test[i:i + 1])[0]
        true = y_test[i]

        print(f"Sample {i + 1}: P(y=1) = {prob:.4f}, "
              f"Predicted: {pred}, True: {true}, "
              f"{'✓' if pred == true else '✗'}")

    # Visualize training
    plot_training_progress(model)

    return model



if __name__ == "__main__":
    demo_sigmoid()
    demo_gradient_cal()
    main()
