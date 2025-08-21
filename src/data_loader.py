from typing import Tuple
import pandas as pd

def load_courses(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if 'avg_rating' in df.columns:
        df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce').fillna(0.0)
    if 'num_ratings' in df.columns:
        df['num_ratings'] = pd.to_numeric(df['num_ratings'], errors='coerce').fillna(0).astype(int)
    return df

def load_ratings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    df['rating'] = df['rating'].astype(float)
    return df

def build_mappings(ratings_df: pd.DataFrame, courses_df: pd.DataFrame):
    users = sorted(ratings_df['username'].unique().tolist())
    user_to_index = {u: i for i, u in enumerate(users)}
    courses = sorted(courses_df['course_id'].unique().tolist())
    course_to_index = {c: i for i, c in enumerate(courses)}
    return user_to_index, course_to_index
