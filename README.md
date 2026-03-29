 🎯 Student Score Prediction using Machine Learning

📌 Overview

This project builds a Linear Regression model to predict a student's math score based on their reading and writing scores.

The goal is to understand how different academic skills relate to each other using real-world data.

---

📊 Dataset

- Dataset: Students Performance Dataset
- Features used:
  - Reading Score
  - Writing Score
- Target:
  - Math Score

---

⚙️ Workflow

1. Loaded dataset using Pandas
2. Explored data using correlation heatmap
3. Selected relevant features
4. Split data into training and testing sets
5. Trained a Linear Regression model
6. Evaluated model performance

---

📈 Model Evaluation

- Mean Absolute Error (MAE) → Measures average prediction error
- R² Score → Measures how well the model explains the data

Interpretation:

- Lower MAE = better accuracy
- Higher R² = better model performance

---

📊 Visualizations

- Feature correlation heatmap
- Actual vs Predicted values plot
- Error distribution plot

---

🔍 Key Insights

- Reading and writing scores show strong correlation with math score
- Writing score often has slightly higher influence on prediction
- Model performs reasonably well but not perfect due to real-world data noise

---

🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

🚀 What I Learned

- Difference between sample data and real-world datasets
- Importance of matching features with dataset
- How to evaluate a machine learning model properly
- How small mistakes in data selection can break the model

---

⚠️ Limitations

- Model is simple (Linear Regression)
- Only 2 input features used
- Does not capture complex relationships

---

🔥 Future Improvements

- Use more features (e.g., gender, lunch, test preparation)
- Try advanced models (Decision Tree, Random Forest)
- Perform feature engineering
- Save and deploy the model

---

📁 Project Structure

Student-Score-Prediction/
│── README.md  
│── student_prediction.py
│── StudentsPerformance.csv

---

👤 Author

Beginner AIML student documenting learning journey 🚀
