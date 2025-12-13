from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
import matplotlib
matplotlib.use('Agg') # Required for server-based plotting (prevents GUI errors)
import matplotlib.pyplot as plt
import io
from fastapi.responses import Response
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import json  # To save the scores

app = FastAPI()

# File paths
MODEL_FILE = "model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
DATA_FILE = r"\Users\Narendra Adnoor\Downloads\salary_prediction-main\data\Employers_data.csv"

# Global variables
model = None
preprocessor = None

def evaluate_model():
    """
    Loads the saved model and calculates accuracy on the test data.
    This runs WITHOUT re-training the model.
    """
    try:
        # 1. Load Data
        data = pd.read_csv(DATA_FILE)
        
        # 2. Split Data (MUST use same random_state as training to get same test set)
        categorical_features = ['Education_Level', 'Job_Title']
        numeric_features = ['Experience_Years']
        
        X = data[numeric_features + categorical_features]
        Y = data['Salary']
        
        # This ensures we test on the exact same rows that were used for testing during training
        _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # 3. Load Model Components
        loaded_model = joblib.load(MODEL_FILE)
        loaded_preprocessor = joblib.load(PREPROCESSOR_FILE)

        # 4. Predict
        X_test_enc = loaded_preprocessor.transform(X_test)
        Y_pred = loaded_model.predict(X_test_enc)

        # 5. Calculate Metrics
        metrics = {
            "Mean_Absolute_Error": round(mean_absolute_error(Y_test, Y_pred), 2),
            "Root_Mean_Squared_Error": round(np.sqrt(mean_squared_error(Y_test, Y_pred)), 2),
            "R2_Score": round(r2_score(Y_test, Y_pred), 4)
        }

        # 6. Save to file (so the endpoint can read it fast)
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        print("\n" + "="*30)
        print("📊 MODEL EVALUATION RESULTS")
        print("="*30)
        print(f"  • R2 Score: {metrics['R2_Score']}")
        print(f"  • MAE:      ${metrics['Mean_Absolute_Error']}")
        print(f"  • RMSE:     ${metrics['Root_Mean_Squared_Error']}")
        print("="*30 + "\n")  
        return metrics

    except Exception as e:
        print(f"Could not evaluate model: {e}")
        return None

def train_model():
    """Train the model, calculate accuracy metrics, and save everything."""
    data = pd.read_csv(DATA_FILE)

    categorical_features = ['Education_Level', 'Job_Title']
    numeric_features = ['Experience_Years']

    # Preprocessor with unknown categories ignored
    preprocessor_local = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Prepare features and target
    X = data[numeric_features + categorical_features]
    Y = data['Salary']

    # Split (optional, just for practice)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Fit preprocessor and model
    X_train_enc = preprocessor_local.fit_transform(X_train)
    model_local = LinearRegression()
    model_local.fit(X_train_enc, Y_train)
    # Calculate Accuracy Parameters
    X_test_enc = preprocessor_local.transform(X_test) # Transform test data
    Y_pred = model_local.predict(X_test_enc)          # Predict on test data

    '''mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)

    metrics = {
        "Mean_Absolute_Error": round(mae, 2),
        "Root_Mean_Squared_Error": round(rmse, 2),
        "R2_Score": round(r2, 4)  # 1.0 is perfect
    }
    # Save metrics to a file so we can read them later
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)'''

    # Save
    joblib.dump(model_local, MODEL_FILE)
    joblib.dump(preprocessor_local, PREPROCESSOR_FILE)
    print("Training complete")
    evaluate_model()

# Load or train model
if os.path.exists(MODEL_FILE) and os.path.exists(PREPROCESSOR_FILE):
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    print("Model and preprocessor loaded.")
    evaluate_model()
else:
    print("Model files not found. Training model...")
    train_model()
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

@app.get("/pred_sal")
def pred_sal(exp: float, edu: str, job: str):
    """Predict salary for a given input."""
    try:
        new_data = pd.DataFrame({
            'Experience_Years': [exp],
            'Education_Level': [edu],
            'Job_Title': [job]
        })

        print("New input data:\n", new_data)

        # Transform input
        new_data_enc = preprocessor.transform(new_data)

        # Convert sparse to dense if needed
        if hasattr(new_data_enc, "toarray"):
            new_data_enc = new_data_enc.toarray()

        print("Encoded input:\n", new_data_enc)

        prediction = model.predict(new_data_enc)
        return {"prediction": float(prediction[0])}

    except Exception as e:
        # Return detailed error
        return {"error": str(e)}
@app.get("/view_graph")
def view_graph():
    """Generates a scatter plot of Experience vs Salary and returns it as an image."""
    try:
        # 1. Load the data again to plot it
        # (Using the global DATA_FILE variable you defined earlier)
        df = pd.read_csv(DATA_FILE)
        
        # 2. Create the plot using Matplotlib
        plt.figure(figsize=(10, 6))
        
        # Scatter plot: x=Experience, y=Salary
        plt.scatter(df['Experience_Years'], df['Salary'], color='blue', alpha=0.6, label='Actual Data')
        
        plt.title('Salary Distribution by Experience')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # 3. Save plot to a memory buffer (RAM) instead of a file
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        
        # 4. Clear the plot to free memory
        plt.close()
        
        # 5. Return the image data
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": f"Could not generate graph: {str(e)}"}
@app.get("/accuracy")
def get_accuracy():
    """Returns the performance metrics of the model."""
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            return json.load(f)
    else:
        return {"error": "Metrics not found. Please delete model.pkl and restart to re-train."}
    
