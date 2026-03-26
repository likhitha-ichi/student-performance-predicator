import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Sleep': [6,7,5,8,6,7,6,5,7,8],
    'Previous_Score': [50,55,60,65,70,72,75,78,80,85],
    'Attendance': [60,65,70,75,80,85,88,90,92,95],
    'Practice_Problems': [10,15,20,25,30,35,40,45,50,60],
    'Marks': [35,40,50,55,65,70,75,80,85,92]
}

df = pd.DataFrame(data)

# Split data
X = df[['Hours', 'Sleep', 'Previous_Score', 'Attendance', 'Practice_Problems']]
y = df['Marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
hours = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
prev = float(input("Enter previous score: "))
att = float(input("Enter attendance (%): "))
practice = float(input("Enter practice problems solved: "))


input_data = pd.DataFrame([[hours, sleep, prev, att, practice]], 
                          columns=['Hours', 'Sleep', 'Previous_Score', 'Attendance', 'Practice_Problems'])

predicted_marks = model.predict(input_data)

print(f"Predicted Marks: {predicted_marks[0]:.2f}")

# Plot
plt.scatter(hours, predicted_marks[0], color='red')
plt.plot(df['Hours'], model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hours vs Marks (Model View)")
plt.show()