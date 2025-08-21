import os, joblib
from .data_loader import load_courses, load_ratings, build_mappings
from .content_model import build_content_model
from .mf_model import MF

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def train_all(courses_csv=None, ratings_csv=None, mf_factors=32, mf_epochs=30):
    if courses_csv is None:
        courses_csv = os.path.join(DATA_DIR, 'courses_sample.csv')
    if ratings_csv is None:
        ratings_csv = os.path.join(DATA_DIR, 'ratings_sample.csv')

    courses_df = load_courses(courses_csv)
    ratings_df = load_ratings(ratings_csv)

    print("Building content model...")
    vect, X = build_content_model(courses_df)

    print("Building mappings...")
    user_to_index, course_to_index = build_mappings(ratings_df, courses_df)
    joblib.dump(user_to_index, os.path.join(MODELS_DIR, 'user_to_index.joblib'))
    joblib.dump(course_to_index, os.path.join(MODELS_DIR, 'course_to_index.joblib'))
    joblib.dump(courses_df, os.path.join(MODELS_DIR, 'courses_df.joblib'))
    joblib.dump(ratings_df, os.path.join(MODELS_DIR, 'ratings_df.joblib'))

    interactions = []
    for _, row in ratings_df.iterrows():
        u = user_to_index[row['username']]; i = course_to_index[row['course_id']]; r = float(row['rating'])
        interactions.append([u, i, r])

    print("Training MF (matrix factorization)...")
    mf = MF(n_users=len(user_to_index), n_items=len(course_to_index), n_factors=mf_factors, n_epochs=mf_epochs)
    mf.fit(interactions)
    mf.save(os.path.join(MODELS_DIR, 'mf_model.joblib'))
    print("Training complete. Models saved to", MODELS_DIR)
