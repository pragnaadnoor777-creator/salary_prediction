import streamlit as st
import requests

st.title("Salary Prediction")

# User input
exp = st.text_input("Experience (years)")
edu = st.selectbox("Education Level",['Bachelor','Master','PhD'])
job = st.selectbox("Job Title",['Engineer','Executive','Intern','Analyst','Manager',''])

if st.button("Predict Salary"):
    try:
        # Call FastAPI endpoint
        response = requests.get(
            "http://127.0.0.1:8001/pred_sal",
            params={"exp": exp, "edu": edu, "job": job}
        )
        result = response.json()

        # Check if prediction exists
        if "prediction" in result:
            st.success(f"Predicted Salary: {result['prediction']:.2f}")
        else:
            st.error(f"Error from API: {result.get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the backend: {e}")

st.write("---") # Adds a visual divider
st.header("Data Visualization")

if st.button("Show Salary Graph"):
    try:
        # Request the image from the backend
        graph_response = requests.get("http://127.0.0.1:8001/view_graph")
        
        if graph_response.status_code == 200:
            # Display the raw image data
            st.image(graph_response.content, caption="Salary vs Experience", use_container_width=True)
        else:
            st.error("Failed to load graph from backend.")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to backend: {e}")
st.write("---")
st.header("Model Performance")

if st.button("Check Model Accuracy"):
    try:
        acc_response = requests.get("http://127.0.0.1:8001/accuracy")
        
        if acc_response.status_code == 200:
            metrics = acc_response.json()
            
            if "error" in metrics:
                st.error(metrics["error"])
            else:
                # Create 3 columns to display metrics side-by-side
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="R2 Score (Accuracy)", value=metrics.get("R2_Score"))
                
                with col2:
                    st.metric(label="Mean Absolute Error", value=f"{metrics.get('Mean_Absolute_Error')}")
                
                with col3:
                    st.metric(label="Root Mean Sq Error", value=f"{metrics.get('Root_Mean_Squared_Error')}")
        else:
            # THIS WILL TELL US THE REAL PROBLEM
            st.error(f"Server Error: {acc_response.status_code}")
            st.write("Server Message:", acc_response.text)
            
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to backend: {e}")



