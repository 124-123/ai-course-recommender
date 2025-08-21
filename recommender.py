from src.mf_model import MF
from src.content_model import load_content_model, content_similarity_for_user_likes
import numpy as np
import pandas as pd
import os

# Path for saving/loading models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


# -----------------------------
# Utility: Blend content + collab scores
# -----------------------------
def blend_scores(content_scores, collab_scores, alpha=0.6):
    content_scores = np.asarray(content_scores, dtype=float)
    collab_scores = np.asarray(collab_scores, dtype=float)
    return alpha * content_scores + (1.0 - alpha) * collab_scores


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_for_user(username, user_to_index, course_to_index, courses_df, ratings_df, alpha=0.6, top_n=10):
    vect, X = load_content_model()

    # Get course list
    course_list = courses_df['course_id'].tolist()

    # Ratings by this user
    user_ratings = ratings_df[ratings_df['username'] == username]

    # Courses user liked
    liked = user_ratings[user_ratings['rating'] >= 4.0]
    liked_indices = [course_list.index(cid) for cid in liked['course_id'].tolist() if cid in course_list]

    # Content-based scores
    content_scores = content_similarity_for_user_likes(liked_indices, X)

    # Collaborative Filtering scores
    mf_path = os.path.join(MODELS_DIR, 'mf_model.joblib')
    collab_scores = np.zeros(len(course_list), dtype=float)
    if os.path.exists(mf_path):
        mf = MF.load(mf_path)
        if username in user_to_index:
            uidx = user_to_index[username]
            collab_scores = mf.predict_user(uidx)
        else:
            collab_scores = mf.global_mean + mf.bi
    else:
        collab_scores = courses_df['avg_rating'].fillna(0.0).astype(float).to_numpy()

    # Final blended score
    final_scores = blend_scores(content_scores, collab_scores, alpha=alpha)

    # Remove already rated courses
    rated_set = set(user_ratings['course_id'].tolist())
    candidates = [(cid, float(score)) for cid, score in zip(course_list, final_scores) if cid not in rated_set]

    # Sort and get top-N
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:top_n]

    # Build results
    recs = []
    for cid, score in top:
        row = courses_df[courses_df['course_id'] == cid].iloc[0].to_dict()
        recs.append({
            'course_id': cid,
            'title': row.get('title'),
            'subject': row.get('subject'),
            'difficulty': row.get('difficulty'),
            'tags': row.get('tags'),
            'score': score
        })
    return recs


# -----------------------------
# Dummy Data for Testing
# -----------------------------
def load_data():
    courses = pd.DataFrame({
        "course_id": [1, 2, 3, 4, 5],
        "title": ["Python Basics", "Data Science", "AI Intro", "Web Dev", "Machine Learning"],
        "avg_rating": [4.2, 4.5, 4.3, 4.1, 4.7],
        "num_ratings": [100, 80, 60, 90, 120],
        "subject": ["CS", "DS", "AI", "Web", "ML"],
        "difficulty": ["Beginner", "Intermediate", "Beginner", "Intermediate", "Advanced"],
        "tags": [["python"], ["data", "science"], ["ai"], ["html", "css"], ["ml", "ai"]],
    })

    ratings = pd.DataFrame({
        "username": ["user1", "user1", "user2", "user2", "user3"],
        "course_id": [1, 2, 2, 3, 5],
        "rating": [5, 4, 4, 3, 5],
    })

    return courses, ratings


# -----------------------------
# Default Alpha
# -----------------------------
def load_alpha():
    return 0.5


# -----------------------------
# Random Baseline Predictor (for demo)
# -----------------------------
def predict_for_user(user_id, all_courses, hist_df, alpha):
    np.random.seed(user_id)  # reproducibility
    scores = np.random.rand(len(all_courses))
    return scores
