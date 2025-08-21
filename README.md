# 🎓 AI-Powered Course Recommendation System

This project is an **AI-powered Course Recommendation System** built with **Python** and **Streamlit**.  
It suggests personalized courses based on user ratings using collaborative filtering techniques.

---

## 🚀 Features
- Upload and use custom **courses** and **ratings** datasets
- Provides **personalized course recommendations**
- Interactive and simple **Streamlit UI**
- Sample datasets included (`courses.csv`, `ratings.csv`)

---

## 📂 Project Structure
```
ai-course-recommender/
│── recommender.py        # Recommendation logic
│── app.py                # Streamlit app file
│── courses.csv           # Sample dataset of courses
│── ratings.csv           # Sample dataset of ratings
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone this repository:
```bash
git clone https://github.com/YOUR-USERNAME/ai-course-recommender.git
cd ai-course-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📊 Sample Datasets

### courses.csv
| course_id | title                          | category        | difficulty   |
|-----------|--------------------------------|----------------|--------------|
| 1         | Introduction to Python         | Programming    | Beginner     |
| 2         | Data Science with Pandas       | Data Science   | Intermediate |
| 3         | Machine Learning Basics        | Machine Learning | Intermediate |
| 4         | Deep Learning with TensorFlow  | Deep Learning  | Advanced     |
| 5         | Natural Language Processing    | AI/NLP         | Advanced     |

### ratings.csv
| user_id | course_id | rating |
|---------|-----------|--------|
| 1       | 1         | 5      |
| 1       | 2         | 4      |
| 2       | 2         | 5      |
| 2       | 3         | 3      |
| 3       | 3         | 4      |
| 3       | 4         | 5      |
| 4       | 5         | 4      |
| 5       | 1         | 3      |

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repo, submit issues, or create pull requests.

---

## 📜 License
This project is open-source and available under the **MIT License**.
"# ai-course-recommender" 
