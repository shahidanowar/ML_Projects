import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load('model_pipeline.pkl')

# Set the title of the app
st.title('House Price Predictor')

st.write("""
### Enter the details of the house to predict its price.
""")

# Create input fields for user
col1, col2 = st.columns(2)

with col1:
    area = st.number_input('Area (in sq. ft.)', min_value=1000, max_value=20000, value=3500, step=100)
    bedrooms = st.selectbox('Bedrooms', [1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox('Bathrooms', [1, 2, 3, 4], index=0)
    stories = st.selectbox('Stories', [1, 2, 3, 4], index=1)
    parking = st.selectbox('Parking Spaces', [0, 1, 2, 3], index=1)

with col2:
    mainroad = st.selectbox('Main Road Access', ['yes', 'no'], index=0)
    guestroom = st.selectbox('Guest Room', ['yes', 'no'], index=1)
    basement = st.selectbox('Basement', ['yes', 'no'], index=1)
    hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'], index=1)
    airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'], index=1)
    prefarea = st.selectbox('Preferred Area', ['yes', 'no'], index=1)
    furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'], index=1)

# Predict button
if st.button('Predict Price'):
    # Create a dataframe from the inputs
    input_data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'parking': parking,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }])

    # Predict the price
    prediction = pipeline.predict(input_data)[0]

    # Display the prediction
    st.success(f'Predicted House Price: ${prediction:,.2f}')
