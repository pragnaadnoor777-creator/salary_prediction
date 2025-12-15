from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg') # Required for server-based plotting (prevents GUI errors)
import matplotlib.pyplot as plt
import io
from fastapi.responses import Response
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import json  # To save the scores
import shap

app = FastAPI()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths (relative)
MODEL_FILE = BASE_DIR / "model.pkl"
PREPROCESSOR_FILE = BASE_DIR / "preprocessor.pkl"
DATA_FILE = BASE_DIR / "data" / "Employers_data.csv"

# Global variables
model = None
preprocessor = None

#======================================================
#                    HASINI'S PART
#======================================================

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

#================================================
#             PRAGNA'S PART
#================================================

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

# Load training data (same file used during training)
train_df = pd.read_csv(DATA_FILE)   

# Keep only input columns
X_train = train_df[["Experience_Years", "Education_Level", "Job_Title"]]

# Encode training data
X_train_enc = preprocessor.transform(X_train)

# Create SHAP explainer WITH background data
explainer = shap.Explainer(model, X_train_enc)

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
        return {"predicted_salary": float(prediction[0])}

    except Exception as e:
        # Return detailed error
        return {"error": str(e)}

#========================================================
#                     PAAVANI'S PART
#========================================================

@app.get("/view_graph")
def view_graph(exp: float, edu: str, job: str):
    try:
        df = pd.read_csv(DATA_FILE)

        # Predict salary again (for safety)
        input_df = pd.DataFrame({
            "Experience_Years": [exp],
            "Education_Level": [edu],
            "Job_Title": [job]
        })

        input_enc = preprocessor.transform(input_df)
        pred_salary = model.predict(input_enc)[0]

        plt.figure(figsize=(10, 6))

        # Actual data
        plt.scatter(
            df['Experience_Years'],
            df['Salary'],
            alpha=0.4,
            label="Actual Data"
        )

        # Predicted point
        plt.scatter(
            exp,
            pred_salary,
            color="red",
            s=120,
            label="Predicted Salary"
        )

        plt.title(
            f"Salary vs Experience\n"
            f"(Education: {edu}, Job: {job})"
        )
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

@app.get("/accuracy")
def get_accuracy():
    """Returns the performance metrics of the model."""
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            return json.load(f)
    else:
        return {"error": "Metrics not found. Please delete model.pkl and restart to re-train."}

#===================================================
#                NIKHITA'S PART
#===================================================

@app.get("/shap_explain")
def shap_explain(exp: float, edu: str, job: str):
    try:
        input_df = pd.DataFrame({
            "Experience_Years": [exp],
            "Education_Level": [edu],
            "Job_Title": [job]
        })

        input_enc = preprocessor.transform(input_df)

        background = preprocessor.transform(
            pd.read_csv(DATA_FILE)[
                ["Experience_Years", "Education_Level", "Job_Title"]
            ].iloc[:50]
        )

        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(input_enc)

        feature_names = preprocessor.get_feature_names_out()

        explanation = [
            {
                "feature": feature_names[i],
                "shap_value": float(shap_values[0][i])
            }
            for i in range(len(feature_names))
        ]

        return {
            "base_value": float(explainer.expected_value),
            "prediction": float(model.predict(input_enc)[0]),
            "shap_values": explanation
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/shap_waterfall")
def shap_waterfall(exp: float, edu: str, job: str):
    try:
        # Input
        input_df = pd.DataFrame({
            "Experience_Years": [exp],
            "Education_Level": [edu],
            "Job_Title": [job]
        })

        input_enc = preprocessor.transform(input_df)

        # Model pieces
        intercept = model.intercept_
        coef = model.coef_
        feature_names = preprocessor.get_feature_names_out()

        # Individual feature contributions
        contributions = {
            feature_names[i]: coef[i] * input_enc[0][i]
            for i in range(len(feature_names))
        }

        # Group contributions
        grouped = {
            "Experience": sum(v for k, v in contributions.items() if "Experience" in k),
            "Education": sum(v for k, v in contributions.items() if "Education_Level" in k),
            "Job Title": sum(v for k, v in contributions.items() if "Job_Title" in k),
        }

        # Waterfall values
        labels = ["Base Salary"] + list(grouped.keys()) + ["Final Prediction"]
        values = [intercept] + list(grouped.values())
        final_prediction = intercept + sum(grouped.values())
        values.append(final_prediction)

        # Cumulative for plotting
        cumulative = [0]
        for v in values[:-1]:
            cumulative.append(cumulative[-1] + v)

        # Plot
        plt.figure(figsize=(9, 5))

        colors = ["gray"] + ["green" if v >= 0 else "red" for v in grouped.values()] + ["blue"]

        for i in range(len(values)):
            plt.barh(
                labels[i],
                values[i],
                left=cumulative[i],
                color=colors[i]
            )

        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("Salary Prediction Waterfall Explanation")
        plt.xlabel("Salary Value")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

