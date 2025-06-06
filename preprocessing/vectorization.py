import os
import joblib
from scipy import sparse

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def save_vectorizer(vectorizer, filename='tfidf_vectorizer.pkl'):
    """Save a trained vectorizer to disk inside /models."""
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(vectorizer, filepath)

def load_vectorizer(filename='tfidf_vectorizer.pkl'):
    """Load a trained vectorizer from disk inside /models."""
    filepath = os.path.join(MODEL_DIR, filename)
    return joblib.load(filepath)

def save_sparse_matrix(matrix, filename='X_tfidf.npz'):
    """Save a sparse matrix to disk inside /models."""
    filepath = os.path.join(MODEL_DIR, filename)
    sparse.save_npz(filepath, matrix)

def load_sparse_matrix(filename='X_tfidf.npz'):
    """Load a sparse matrix from disk inside /models."""
    filepath = os.path.join(MODEL_DIR, filename)
    return sparse.load_npz(filepath)
