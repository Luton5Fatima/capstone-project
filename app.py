import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import openai

# Load the trained model
model = joblib.load("rf_model.pkl")

# Function to query GPT
def query_gpt(item_name, material, prediction_date, predicted_rate):
    openai.api_key = "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"  # Replace with your OpenAI API key

    response = openai.Completion.create(
        model="text-davinci-003",  # or the latest model version
        prompt=f"Analyze the predicted rate for {item_name}, material code {material}, on {prediction_date}. The predicted rate is {predicted_rate:.2f}. Provide insights on market trends, potential alternatives, and whether the rate is reasonable for a large oil and gas organization using web for latest research.",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Streamlit app
def main():
    st.title("Rate Prediction and Analysis App")

    # Get user inputs
    item_name = st.text_input("Enter Item Name:")
    material = st.text_input("Enter Material/Service Code:")
    prediction_date = st.date_input("Select Prediction Date:")
    
    if st.button("Predict"):
        # Validate input material
        # [List of valid materials] - Replace with your actual list of materials
        valid_materials = [30177041, 30177043, 30177045]  # Example list, add your full list here
        
        try:
            material = int(material)
            if material not in valid_materials:
                st.error("Invalid Material/Service Code. Please enter a valid code.")
                return
        except ValueError:
            st.error("Invalid Material/Service Code. Please enter a valid code.")
            return
    
        # Prepare input data for prediction
        input_data = {
            "Material/Service": [material],
            "Year": [prediction_date.year],
            "Month": [prediction_date.month],
            "Lagged_Rate": [1800]  # Replace with your desired constant value
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Predict rate
        predicted_rate = model.predict(input_df)
        
        # Display prediction result
        st.write(f"Predicted Rate for {item_name} (Material {material}) on {prediction_date}: {predicted_rate[0]:.2f}")

        # Query GPT for analysis
        gpt_analysis = query_gpt(item_name, material, prediction_date, predicted_rate[0])
        st.write("Generative AI Says:")
        st.write(gpt_analysis)

if __name__ == "__main__":
    main()
