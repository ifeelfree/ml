import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
# Try one of these:
matplotlib.use('TkAgg')    # Tkinter backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

word_vectors = {
    'Athens': np.array([0.7, 0.2, 0.1]),
    'Greece': np.array([0.8, 0.3, 0.2]),
    'Baghdad': np.array([0.6, 0.1, 0.4]),
    'Iraq': np.array([0.7, 0.2, 0.5]),
    'Cairo': np.array([0.6, 0.3, 0.3]),
    'Egypt': np.array([0.7, 0.4, 0.4])
}
def cosine_sim(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

def demo_cosine_sim():
    vec1 = np.array([[1, 2, 3], [10, 20, 30]])
    similarity_matrix = cosine_sim(vec1[0], vec1[1])
    print(similarity_matrix)

    vec1 = np.array([[1, -2, 3], [10, 20, 30]])
    similarity_matrix = cosine_sim(vec1[0], vec1[1])
    print(similarity_matrix)

def demo_find_country():

    predicted_country, similarity = find_country('Athens', 'Greece', 'Baghdad', word_vectors)
    print(f"Predicted country for Baghdad: {predicted_country} (similarity: {similarity:.3f})")

def find_country(city1, country1, city2, word_vectors):
    exclude = {city1, country1, city2}
    vec = word_vectors[country1] - word_vectors[city1] + word_vectors[city2]
    best_word = None
    best_score = -1
    for word, wvec in word_vectors.items():
        if word in exclude:
            continue
        score = cosine_sim(vec, wvec)
        if score > best_score:
            best_score = score
            best_word = word
    return best_word, best_score


def compute_pca(X, n_components=2):
    X_demeaned = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
    idx_sorted = np.argsort(eigen_vals)[::-1]
    eigen_vecs_subset = eigen_vecs[:, idx_sorted[:n_components]]
    X_reduced = np.dot(X_demeaned, eigen_vecs_subset)
    return X_reduced

def sklean_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca_sklearn = pca.fit_transform(X)
    return X_pca_sklearn

def demo_compute_pca():
    words = list(word_vectors.keys())
    X = np.array([word_vectors[word] for word in words])
    X_pca_numpy = compute_pca(X, n_components=2)
    X_pca_sklearn = sklean_pca(X, n_components=2)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pca_numpy[:, 0], X_pca_numpy[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, (X_pca_numpy[i, 0], X_pca_numpy[i, 1]))
    plt.title("Native NumPy PCA")

    plt.subplot(1, 2, 2)
    plt.scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, (X_pca_sklearn[i, 0], X_pca_sklearn[i, 1]))
    plt.title("scikit-learn PCA")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_cosine_sim()
    demo_find_country()
    demo_compute_pca()
