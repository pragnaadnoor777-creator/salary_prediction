import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv("//Users//nikhita//salary_prediction//data//Employers_data.csv")
df = pd.DataFrame(data)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([df, one_hot_df], axis=1)

df_encoded = df_encoded.drop(categorical_columns, axis=1)
print(f"Encoded Employee data : \n{df_encoded}")

X = df_encoded[['Experience_Years','Gender']]
Y = df_encoded['Salary']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0.42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
x=model.predict(pd.DataFrame({'Experience_Years': [6]},{'Gender': '24'}))
print(x)
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse**0.5
r2 = r2_score(Y_test, Y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Accuracy Percentage:", r2 * 100, "%")
