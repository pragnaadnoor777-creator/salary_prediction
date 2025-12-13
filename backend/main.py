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

app = FastAPI()

# File paths
MODEL_FILE = "model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
DATA_FILE = r"\Users\Narendra Adnoor\Downloads\salary_prediction-main\data\Employers_data.csv"

# Global variables
model = None
preprocessor = None

def train_model():
    """Train the model and save preprocessor + model."""
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

    # Save
    joblib.dump(model_local, MODEL_FILE)
    joblib.dump(preprocessor_local, PREPROCESSOR_FILE)
    print("Training complete and files saved.")

# Load or train model
if os.path.exists(MODEL_FILE) and os.path.exists(PREPROCESSOR_FILE):
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    print("Model and preprocessor loaded.")
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
    
