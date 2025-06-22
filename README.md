# creditcard_fraudshield

A simple web application to detect fraudulent credit card transactions using a trained machine learning model, built with Flask and Random Forest. The user inputs 30 PCA-transformed transaction features and gets a prediction indicating whether the transaction is normal or fraud.

Project Overview :

This project aims to detect fraudulent credit card transactions using a machine learning model trained on a real-world dataset. The model is deployed via a Flask web application where users can input transaction data and receive instant predictions.

Features :

Accepts user input of 30 PCA-transformed numerical features (comma-separated).
Classifies transactions as:
NORMA
FRAUD
Real-time results displayed in a color-coded format (green for normal, red for fraud).

Model Training (Python) :

Model: RandomForestClassifier
Dataset balancing: 492 fraud + 492 randomly sampled normal transactions
Accuracy achieved on balanced test set is printed in console
Model saved using joblib as fraud_model.pkl

Web Application (Flask):

Route /: Home page with input form
Route /predict: Handles prediction and displays result
Users input 30 comma-separated float values representing a transaction, and receive the prediction: either FRAUD or NORMAL.




