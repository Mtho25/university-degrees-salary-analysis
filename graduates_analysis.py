# Data Science Phase: ML Prediction (Predict Salary by Major)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the graduate data from the 'graduates.csv' file into a pandas DataFrame.
df = pd.read_csv('graduates.csv')  # Using the attached file name

# Define the features (X) that will be used to predict the salary.
# These include 'Year', 'Education.Major', and various demographic indicators.
X = df[['Year', 'Education.Major', 'Demographics.Gender.Females', 'Demographics.Gender.Males', 'Demographics.Ethnicity.Asians', 'Demographics.Ethnicity.Minorities', 'Demographics.Ethnicity.Whites']]
# Define the target variable (y), which is the mean salary we want to predict.
y = df['Salaries.Mean']  # Predicting the mean salary

# Set up a preprocessor for handling categorical features.
# OneHotEncoder will convert the 'Education.Major' column into a numerical format,
# making it suitable for the machine learning model.
# 'handle_unknown="ignore"' ensures the model can handle majors not seen during training.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Education.Major'])
    ], remainder='passthrough') # Keep other numerical columns as they are

# Create a machine learning pipeline.
# First, the preprocessor transforms the data, then a RandomForestRegressor model
# is applied for predicting salaries. RandomForest is an ensemble learning method
# known for its good performance.
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Split the dataset into training and testing sets.
# 80% of the data will be used for training the model, and 20% for evaluating its performance.
# 'random_state' ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regression model using the training data.
model.fit(X_train, y_train)

# Make predictions on the unseen test data.
preds = model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE).
# MSE measures the average squared difference between the estimated values and the actual values.
mse = mean_squared_error(y_test, preds)
print(f'Mean Squared Error (MSE): {mse}')

# Data Analysis Phase: SQL (Using SQLite) - Analyzing Graduate Data with SQL Queries
import sqlite3
import pandas as pd

# Load the existing pandas DataFrame (df) into an in-memory SQLite database.
# This allows us to interact with our dataset using standard SQL queries.
# The table will be named 'grads', and the DataFrame index will not be stored.
conn = sqlite3.connect(':memory:')
df.to_sql('grads', conn, index=False)

# Define an SQL query to calculate the average salary for each 'Education.Major'.
# The results are grouped by major and ordered in descending order of average salary.
query = """
SELECT "Education.Major", AVG("Salaries.Mean") AS Avg_Salary
FROM grads
GROUP BY "Education.Major"
ORDER BY Avg_Salary DESC
"""

# Execute the SQL query and load the returned data into a new pandas DataFrame called 'results'.
results = pd.read_sql(query, conn)
print(results)

# For Power BI Integration: Export the original DataFrame to a CSV file.
# This file can then be easily imported into Power BI to create interactive
# visualizations and perform further detailed data analysis.
df.to_csv('grad_intel.csv', index=False)
