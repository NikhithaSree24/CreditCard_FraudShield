import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('creditcard.csv')

# Balance the dataset: 492 fraud + 492 normal
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0].sample(n=492, random_state=42)
new_df = pd.concat([fraud, normal])

# Features and target
X = new_df.drop(columns='Class')
y = new_df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate (optional print)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Save model
joblib.dump(model, 'fraud_model.pkl')
print("Model saved as fraud_model.pkl")
