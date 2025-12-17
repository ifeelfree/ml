import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

def calculate_w(X, Y, method='linear_regression'):
    if method == 'linear_regression':
        return calcualte_w_linear_regression(X, Y)
    elif method == 'gradient_descent':
        return calculate_w_gradient_sescent(X, Y)
    else:
        return calculate_w_closed_form(X, Y)

def calculate_w_closed_form(X, Y):
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    # or
    # W = np.linalg.pinv(X)@Y
    return W

def calcualte_w_linear_regression(X, Y):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    W = reg.coef_
    return W

def calculate_w_gradient_sescent(X, Y, learning_rate = 0.01, n_epochs=100000):
    # Initialize W randomly
    np.random.seed(42)
    W = np.random.randn(3, 3)


    for epoch in range(n_epochs):
        # Forward pass
        Y_pred = X @ W
        # Compute loss (mean squared error)
        loss = np.mean((Y_pred - Y) ** 2)
        # Compute gradient
        grad_W = 2 * X.T @ (Y_pred - Y) / X.shape[0]
        # Update W
        W -= learning_rate * grad_W
        # Optionally print loss
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    return W


if __name__ == "__main__":

    # 1. Load pre-trained embeddings (dummy example, replace with real embeddings)
    # Suppose each word is represented by a 3D vector for simplicity
    english_embeddings = {
        'cat': np.array([0.1, 0.3, 0.5]),
        'dog': np.array([0.2, 0.4, 0.6]),
        'house': np.array([0.9, 0.1, 0.3]),
        'car': np.array([0.5, 0.7, 0.2]),
        'tree': np.array([0.6, 0.8, 0.4]),
        'book': np.array([0.3, 0.2, 0.9]),
        'water': np.array([0.7, 0.5, 0.1]),
        'sun': np.array([0.8, 0.6, 0.2]),
        'apple': np.array([0.4, 0.9, 0.3]),
        'school': np.array([0.2, 0.8, 0.7])
    }
    french_embeddings = {
        'chat': np.array([0.11, 0.29, 0.52]),
        'chien': np.array([0.21, 0.39, 0.61]),
        'maison': np.array([0.88, 0.12, 0.31]),
        'voiture': np.array([0.51, 0.69, 0.19]),
        'arbre': np.array([0.59, 0.81, 0.41]),
        'livre': np.array([0.29, 0.21, 0.92]),
        'eau': np.array([0.72, 0.48, 0.13]),
        'soleil': np.array([0.81, 0.62, 0.18]),
        'pomme': np.array([0.39, 0.88, 0.29]),
        'école': np.array([0.19, 0.82, 0.69])
    }

    # 2. Prepare bilingual dictionary
    english_words = [
        'cat', 'dog', 'house', 'car', 'tree',
        'book', 'water', 'sun', 'apple', 'school'
    ]
    french_words = [
        'chat', 'chien', 'maison', 'voiture', 'arbre',
        'livre', 'eau', 'soleil', 'pomme', 'école'
    ]

    X = np.array([english_embeddings[w] for w in english_words])
    Y = np.array([french_embeddings[w] for w in french_words])

    # 3. Learn linear transformation

    W = calculate_w(X,Y, )


    # 4. Translate new word
    def translate(word, english_embeddings, french_embeddings, W):
        if word not in english_embeddings:
            return "Word not in English embeddings."
        # Transform English embedding to French space
        transformed = np.dot(english_embeddings[word], W.T)
        # Find closest French word by cosine similarity
        french_words = list(french_embeddings.keys())
        french_vecs = np.array([french_embeddings[w] for w in french_words])
        sims = cosine_similarity([transformed], french_vecs)[0]
        best_idx = np.argmax(sims)
        return french_words[best_idx]

    # Example translation
    english_word = 'cat'
    french_translation = translate(english_word, english_embeddings, french_embeddings, W)
    print(f"English: {english_word} -> French: {french_translation}")