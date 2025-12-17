from sklearn.feature_extraction.text import CountVectorizer

def demo_bag_of_words():
    # Sample documents
    documents = [
        "I love machine learning",
        "Machine learning is fun",
        "I love coding"
    ]

    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the documents
    X = vectorizer.fit_transform(documents)

    # Show the feature names (unique words)
    print("Feature names:", vectorizer.get_feature_names_out())

    # Show the bag-of-words representation (word counts)
    print("Bag-of-words matrix:\n", X.toarray())



import numpy as np

class LSH:
    def __init__(self, n_planes=10, n_tables=5, dim=3):
        self.n_planes = n_planes
        self.n_tables = n_tables
        self.dim = dim
        # Each table has its own set of random hyperplanes
        self.planes = [np.random.randn(n_planes, dim) for _ in range(n_tables)]
        self.tables = [{} for _ in range(n_tables)]

    def _hash(self, v, planes):
        # Project v onto each hyperplane: sign(dot) gives 0 or 1
        return tuple((v @ planes.T) > 0)

    def add(self, vec_id, v):
        for t in range(self.n_tables):
            h = self._hash(v, self.planes[t])
            if h not in self.tables[t]:
                self.tables[t][h] = []
            self.tables[t][h].append((vec_id, v))

    def query(self, v, top_k=1):
        candidates = set()
        for t in range(self.n_tables):
            h = self._hash(v, self.planes[t])
            bucket = self.tables[t].get(h, [])
            for vec_id, vec in bucket:
                candidates.add((vec_id, tuple(vec)))
        # Remove duplicates and compute cosine similarity
        candidate_list = []
        for vec_id, vec_tuple in candidates:
            vec = np.array(vec_tuple)
            sim = np.dot(v, vec) / (np.linalg.norm(v) * np.linalg.norm(vec))
            candidate_list.append((vec_id, sim))
        # Sort by similarity
        candidate_list.sort(key=lambda x: -x[1])
        return candidate_list[:top_k]

if __name__ == '__main__':
    demo_bag_of_words()
    # Suppose we have 5 vectors in 3D
    data = {
        'a': np.array([0.1, 0.3, 0.5]),
        'b': np.array([0.2, 0.4, 0.6]),
        'c': np.array([0.9, 0.1, 0.3]),
        'd': np.array([0.5, 0.7, 0.2]),
        'e': np.array([0.6, 0.8, 0.4])
    }

    lsh = LSH(n_planes=5, n_tables=3, dim=3)
    for key, vec in data.items():
        lsh.add(key, vec)

    # Query for nearest neighbor to a new vector
    query_vec = np.array([0.15, 0.35, 0.55])
    neighbors = lsh.query(query_vec, top_k=2)
    print("Approximate nearest neighbors:", neighbors)