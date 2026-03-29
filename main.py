import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import seaborn as sns 
df = pd.read_csv('StudentsPerformance.csv') 
print(df.columns)
sns.heatmap(df.corr(numeric_only=True), annot=True)
print(df[['math score', 'reading score', 'writing score']].corr())
plt.title("Feature Correlation")
plt.show()


# Split data
X = df[['reading score', 'writing score']]
y = df['math score']
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
print("\nModel Interpretation:")
print(f"On average, predictions are off by {mae:.2f} marks.")
print(f"Model explains {r2*100:.2f}% of the variation in marks.")
#Feature importance

print("\nFeature importance:")
for i,col in enumerate(X.columns):
  print(f"{col}:{model.coef_[i]:.2f}")
# Prediction
try:
    reading = float(input("Enter reading score: "))
    writing = float(input("Enter writing score: "))

    input_data = pd.DataFrame([[reading, writing]],
                              columns=['reading score', 'writing score'])

    predicted_marks = model.predict(input_data)

    print(f"Predicted Math Score: {predicted_marks[0]:.2f}")

except:
    print("Invalid input. Enter numbers only.")
# Plot
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, color='red', label='Predicted Points')

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--', color='blue', label='Perfect Prediction')

plt.xlabel("Actual Math score")
plt.ylabel("Predicted math score")
plt.title("Actual vs Predicted")

plt.legend()
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(y_test - y_pred, kde=True)
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.show()
