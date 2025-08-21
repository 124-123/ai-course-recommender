import argparse, os, joblib
from .train import train_all
from ..recommender import recommend_for_user

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def train(args):
    train_all(courses_csv=args.courses, ratings_csv=args.ratings, mf_factors=args.factors, mf_epochs=args.epochs)

def recommend(args):
    user_to_index = joblib.load(os.path.join(MODELS_DIR, 'user_to_index.joblib'))
    course_to_index = joblib.load(os.path.join(MODELS_DIR, 'course_to_index.joblib'))
    courses_df = joblib.load(os.path.join(MODELS_DIR, 'courses_df.joblib'))
    ratings_df = joblib.load(os.path.join(MODELS_DIR, 'ratings_df.joblib'))

    recs = recommend_for_user(args.username, user_to_index, course_to_index, courses_df, ratings_df, alpha=args.alpha, top_n=args.top)
    if not recs:
        print("No recommendations found.")
    else:
        print(f"Top {len(recs)} recommendations for {args.username}:")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['title']} ({r['course_id']}) — score: {r['score']:.4f} — {r['subject']} / {r['difficulty']}")

def main():
    parser = argparse.ArgumentParser(prog="Personalized Recommender")
    sub = parser.add_subparsers(dest='cmd')
    t = sub.add_parser('train')
    t.add_argument('--courses', default=None)
    t.add_argument('--ratings', default=None)
    t.add_argument('--factors', type=int, default=32)
    t.add_argument('--epochs', type=int, default=30)
    r = sub.add_parser('recommend')
    r.add_argument('username')
    r.add_argument('--alpha', type=float, default=0.6)
    r.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'recommend':
        recommend(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
