import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('placement.csv')

# Extract features and target variable
x = df[['cgpa']]
y = df['package']

# Create and fit the Linear Regression model
lr = LinearRegression()
lr.fit(x, y)

# Function to calculate and display the prediction
def predict_package():
    user_cgpa = float(cgpa_entry.get())
    predicted_package = lr.predict([[user_cgpa]])[0]
    result_label.config(text=f"Predicted Package: {predicted_package:.2f} LPA")

# Create the main window
root = tk.Tk()
root.title("Package Prediction")

# Create and configure a frame
frame = ttk.Frame(root, padding=10)
frame.grid(column=0, row=0)

# CGPA Label and Entry
cgpa_label = ttk.Label(frame, text="Enter CGPA:")
cgpa_label.grid(column=0, row=0, padx=5, pady=5)

cgpa_entry = ttk.Entry(frame)
cgpa_entry.grid(column=1, row=0, padx=5, pady=5)

# Prediction Button
predict_button = ttk.Button(frame, text="Predict", command=predict_package)
predict_button.grid(column=0, row=1, columnspan=2, padx=5, pady=10)

# Result Label
result_label = ttk.Label(frame, text="")
result_label.grid(column=0, row=2, columnspan=2, padx=5, pady=5)

# Start the Tkinter main loop
root.mainloop()
