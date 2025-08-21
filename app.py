import streamlit as st
import pandas as pd
from recommender import load_alpha, predict_for_user

st.set_page_config(page_title="AI Course Recommender", layout="wide")
st.title("🎓 AI-Powered Course Recommendation System")

# --- File Upload Section ---
st.sidebar.header("📂 Upload Data")

uploaded_courses = st.sidebar.file_uploader("Upload Courses CSV", type=["csv"])
uploaded_ratings = st.sidebar.file_uploader("Upload Ratings CSV", type=["csv"])

if uploaded_courses is not None and uploaded_ratings is not None:
    courses = pd.read_csv(uploaded_courses)
    ratings = pd.read_csv(uploaded_ratings)

    # Validate required columns
    if not {"course_id", "title", "avg_rating", "num_ratings"}.issubset(courses.columns):
        st.error("❌ Courses CSV must contain: course_id, title, avg_rating, num_ratings")
    elif not {"user", "course", "rating"}.issubset(ratings.columns):
        st.error("❌ Ratings CSV must contain: user, course, rating")
    else:
        alpha = load_alpha()

        # Sidebar - User selection
        user_id = st.sidebar.selectbox("Select a User ID", ratings["user"].unique())

        # Show user history
        user_history = ratings[ratings["user"] == user_id]
        st.subheader("📚 Your Course History")
        if not user_history.empty:
            st.dataframe(user_history.rename(columns={"course": "Course ID", "rating": "Your Rating"}))
        else:
            st.info("No rating history found. Cold-start recommendations will be shown.")

        # --- Recommendation Logic ---
        st.subheader("✨ Recommended Courses for You")

        if user_history.empty:
            # Cold start → Top courses by avg rating
            top_courses = courses.sort_values(by=["avg_rating", "num_ratings"], ascending=False).head(10)
            recs = [{"course": row["course_id"], "score": row["avg_rating"]} for _, row in top_courses.iterrows()]
        else:
            all_courses = courses["course_id"].tolist()
            scores = predict_for_user(user_id, all_courses, user_history, alpha)

            rated_cids = set(user_history["course"].tolist())
            scored = [(cid, float(s)) for cid, s in zip(all_courses, scores) if cid not in rated_cids]
            scored.sort(key=lambda x: x[1], reverse=True)

            recs = [{"course": cid, "score": sc} for cid, sc in scored[:10]]

        # --- Display Recommendations ---
        for rec in recs:
            course_row = courses[courses["course_id"] == rec["course"]].iloc[0]
            st.write(f"**{course_row['title']}**  |  ⭐ {round(rec['score'], 2)}")

else:
    st.warning("⬅️ Please upload both **Courses CSV** and **Ratings CSV** to continue.")
