import csv
import joblib

try:
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    raise ImportError('scikit-learn is required to run this script. Install with pip install scikit-learn') from e

# Load training data
with open('training_data.csv', newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)

if not rows:
    raise ValueError('training_data.csv is empty')

# Assume the first row is a header
data_rows = rows[1:]
X = [list(map(float, row[:-1])) for row in data_rows]
y = [float(row[-1]) for row in data_rows]

# Train linear model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'linear_model.pkl')

# Save model coefficients to text file
with open('linear_model.txt', 'w') as f:
    f.write(f'Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n')
