import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Medical_insurance.csv')

# Removing duplicate entries
data.drop_duplicates(inplace=True)

# Filling missing values
data['bmi'].fillna(data['bmi'].median(), inplace=True)
data['smoker'].fillna(data['smoker'].mode()[0], inplace=True)

# Creating BMI category
data['BMI_Category'] = pd.cut(data['bmi'], bins=[0, 18.5, 25, 30, float('inf')],
                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Converting categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['smoker', 'region', 'sex'])

# Normalizing the 'Age' and 'BMI' columns using StandardScaler
scaler = StandardScaler()
data[['age', 'bmi']] = scaler.fit_transform(data[['age', 'bmi']])

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting the distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], kde=True, color='blue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plotting the distribution of BMI
plt.figure(figsize=(10, 6))
sns.histplot(data['bmi'], kde=True, color='green')
plt.title('Distribution of BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# Excluding 'BMI_Category' before computing the correlation matrix
correlation_data = data.drop(columns=['BMI_Category'])  # Excluding the categorical column
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Boxplot for Charges by Smoker status
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['smoker_yes'], y=data['charges'])
plt.title('Insurance Costs by Smoking Status')
plt.xlabel('Smoker')
plt.ylabel('Charges')
plt.show()

# Boxplot for Charges by Region
plt.figure(figsize=(10, 6))
sns.boxplot(x='region_northwest', y='charges', data=data)
plt.title('Insurance Costs by Region')
plt.xlabel('Region')
plt.ylabel('Charges')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data['charges'])
plt.title('Boxplot for Charges')
plt.xlabel('charges')
plt.show()



# Model Development

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Encoding categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Separating the features and target variable
X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_predictions = lr.predict(X_test_scaled)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_predictions = rf.predict(X_test_scaled)

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_predictions = gb.predict(X_test_scaled)

# Evaluating the models
lr_mse = mean_squared_error(y_test, lr_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)

print(f"Linear Regression MSE: {lr_mse}")
print(f"Random Forest MSE: {rf_mse}")
print(f"Gradient Boosting MSE: {gb_mse}")



import matplotlib.pyplot as plt

# Data for plotting
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
mse_values = [36566257.199102886, 21801949.932984788, 17714020.229449302]

plt.figure(figsize=(10, 5))
plt.bar(models, mse_values, color=['blue', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('Comparison of Model Performance (MSE)')
plt.ylim([0, 40000000])  # Setting the limit for better relative comparison
plt.show()
