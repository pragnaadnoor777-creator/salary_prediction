import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv("//Users//nikhita//salary_prediction//data//Employers_data.csv")
df = pd.DataFrame(data)
#print(df)
categorical_features = ['Education_Level'] 
numeric_features = ['Experience_Years']     

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)


X = df[["Experience_Years","Education_Level"]]
Y = df["Salary"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)

model = LinearRegression()
model.fit(X_train_enc, Y_train)

new_pred = pd.DataFrame({
'Experience_Years' : [6],
'Education_Level' : ['Master']})
new_pred_enc = preprocessor.transform(new_pred)
Y_pred = model.predict(new_pred_enc)
print(Y_pred)
