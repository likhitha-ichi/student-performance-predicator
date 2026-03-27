 import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score

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

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

#Train model
model = LinearRegression()
model.fit(X_train,y_train)

#Evaluation
y_pred = model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("mean Absolute Error:",mae)
print("R2 score:",r2)

#Feature importance

print("\nFeature importance:")
for i,col in enumerate(X.columns):
  print(f"{col}:{model.coef_[i]:.2f}")
# Prediction
try:
    hours = float(input("Enter study hours: "))
    sleep = float(input("Enter sleep hours: "))
    prev = float(input("Enter previous score: "))
    att = float(input("Enter attendance (%): "))
    practice = float(input("Enter practice problems solved: "))


    input_data = pd.DataFrame([[hours, sleep, prev, att, practice]],
                          columns=['Hours', 'Sleep', 'Previous_Score', 'Attendance', 'Practice_Problems'])

    predicted_marks = model.predict(input_data)

    print(f"Predicted Marks: {predicted_marks[0]:.2f}")
except:
  print("invalid input.enter numbers only")

# Plot
y_pred = model.predict(X)

plt.scatter(df['Marks'], y_pred,color='red',label='predicted points')

plt.plot([df['Marks'].min(), df['Marks'].max()],
         [df['Marks'].min(), df['Marks'].max()],
         linestyle='--',color='blue',label='perfect predicted line')

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted")
for actual, pred in zip(df['Marks'], y_pred):
    print(f"Actual: {actual}, Predicted: {pred}")

plt.legend
plt.show()
