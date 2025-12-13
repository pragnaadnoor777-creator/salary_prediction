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
            "http://127.0.0.1:8000/pred_sal",
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
        graph_response = requests.get("http://127.0.0.1:8000/view_graph")
        
        if graph_response.status_code == 200:
            # Display the raw image data
            st.image(graph_response.content, caption="Salary vs Experience", use_column_width=True)
        else:
            st.error("Failed to load graph from backend.")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to backend: {e}")




