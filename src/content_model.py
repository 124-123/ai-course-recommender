from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib, numpy as np, os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
MATRIX_PATH = os.path.join(MODELS_DIR, 'tfidf_matrix.joblib')

def build_content_model(courses_df, text_cols=('title','subject','tags'), max_features=5000):
    corpus = (courses_df.get(text_cols[0], '').fillna('') + ' ')
    for col in text_cols[1:]:
        corpus = corpus + ' ' + courses_df.get(col, '').fillna('')
    corpus = corpus.str.lower().tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1,2))
    X = vectorizer.fit_transform(corpus)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(X, MATRIX_PATH)
    return vectorizer, X

def load_content_model():
    vect = joblib.load(VECTORIZER_PATH)
    X = joblib.load(MATRIX_PATH)
    return vect, X

def content_similarity_for_user_likes(liked_item_indices, X):
    if len(liked_item_indices) == 0:
        return np.zeros(X.shape[0], dtype=float)
    user_profile = X[liked_item_indices].mean(axis=0)
    if hasattr(user_profile, "toarray"):
        user_profile = user_profile.toarray()
    user_profile = np.asarray(user_profile).reshape(1, -1)
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)
    sims = cosine_similarity(user_profile, X_dense).ravel()
    return sims
