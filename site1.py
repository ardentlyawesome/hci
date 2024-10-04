import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model

# Custom CSS to improve styling
# Dark mode CSS styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;  /* Dark background */
        color: #ffffff;  /* Light text color */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffcc00;  /* Gold color for headers */
    }
    .stButton {
        background-color: #ffcc00;  /* Gold button color */
        color: #1e1e1e;  /* Dark text color on buttons */
    }
    .stLineChart {
        background-color: #1e1e1e;  /* Chart background */
        color: #ffffff;  /* Chart text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load and preprocess data (for Vanilla LSTM, for example)
def load_and_preprocess_data():
    column_names = ["Date", "OilPrice"]
    brent_data = pd.read_excel("https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls", sheet_name=1, skiprows=4, engine='xlrd', names=column_names)
    brent_data.dropna(inplace=True)  # Drop missing values if any
    brent_data['Date'] = pd.to_datetime(brent_data['Date'])
    brent_data.set_index('Date', inplace=True)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    brent_data['normalized_oil_prices'] = scaler.fit_transform(brent_data[['OilPrice']])
    
    return brent_data, scaler

# Function to load Vanilla LSTM model
def load_lstm_model():
    model = load_model('C:/Users/aindr/OneDrive/Documents/Stuff/hci/vanilla.h5')  # Update path as needed
    return model

# Load Stacked LSTM Model
def load_stacked_lstm_model():
    model = load_model('stacked_lstm_oil_price_model.h5')  # Update with the actual path to your Stacked LSTM model
    return model

# Load Bidirectional LSTM Model
def load_bidirectional_lstm_model():
    model = load_model('bi_di_LSTM.h5')  # Update with the actual path to your Bidirectional LSTM model
    return model

# Function to make predictions with Vanilla LSTM model
def make_predictions(model, test_data):
    # Reshape the test data for predictions
    test_data_reshaped = np.reshape(test_data.values, (test_data.shape[0], 1, 1))  # Reshape as per model input
    predictions = model.predict(test_data_reshaped)
    return predictions

# Visualization function for model
def plot_predictions(test_data, predictions, start_date, end_date):
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data.values, label="Actual")
    plt.plot(test_data.index, predictions, label="Predicted")
    
    # Set x-axis limits to the specified date range
    plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    
    plt.title("Model Predictions")
    plt.xlabel("Time")
    plt.ylabel("Normalized Oil Prices")
    plt.legend()
    st.pyplot(plt)

# Page 4: Vanilla LSTM Model
def vanilla_lstm_model():
    st.title("Vanilla LSTM Model")
    st.write("Predictions and metrics for the Vanilla LSTM model.")

    # Load and preprocess the data
    brent_data, scaler = load_and_preprocess_data()
    
    # Load the Vanilla LSTM model
    model = load_lstm_model()

    # Make predictions
    predictions = make_predictions(model, brent_data['normalized_oil_prices'])

    # Date range input
    st.write("Select Date Range:")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

    # Filter data based on date range
    filtered_test_data = brent_data.loc[start_date:end_date]['normalized_oil_prices']
    filtered_predictions = predictions[brent_data.index.get_indexer(filtered_test_data.index)]

    # Plot the predictions vs actual values within the selected date range
    plot_predictions(filtered_test_data, filtered_predictions, start_date, end_date)


# Page 1: OSLR Model
def oslr_model():
    st.title("OSLR Model")
    st.write("Description of OSLR model here.")
    st.line_chart([1, 2, 3, 4])  # Dummy chart

# Page 2: Stacked LSTM Model
def stacked_lstm_model():
    st.title("Stacked LSTM Model")
    st.write("Description of Stacked LSTM model here.")
    
    # Load data and model
    brent_data, scaler = load_and_preprocess_data()
    model = load_stacked_lstm_model()
    
    # Make predictions
    predictions = make_predictions(model, brent_data['normalized_oil_prices'])
    
    # Date range input
    st.write("Select Date Range:")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
         
    # Filter data based on date range
    filtered_test_data = brent_data.loc[start_date:end_date]['normalized_oil_prices']
    filtered_predictions = predictions[brent_data.index.get_indexer(filtered_test_data.index)]

    # Plot the predictions vs actual values within the selected date range
    plot_predictions(filtered_test_data, filtered_predictions, start_date, end_date)

# Page 3: Bidirectional LSTM Model
def bidirectional_lstm_model():
    st.title("Bidirectional LSTM Model")
    st.write("Description of Bidirectional LSTM model here.")
    
    # Load data and model
    brent_data, scaler = load_and_preprocess_data()
    model = load_bidirectional_lstm_model()
    
    # Make predictions
    predictions = make_predictions(model, brent_data['normalized_oil_prices'])
    
    # Date range input
    st.write("Select Date Range:")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    
    # Filter data based on date range
    filtered_test_data = brent_data.loc[start_date:end_date]['normalized_oil_prices']
    filtered_predictions = predictions[brent_data.index.get_indexer(filtered_test_data.index)]

    # Plot the predictions vs actual values within the selected date range
    plot_predictions(filtered_test_data, filtered_predictions, start_date, end_date)

# Streamlit tab navigation
def main():
    st.sidebar.title("Choose the model you want to visualize")
    option = st.sidebar.selectbox("Select Model", ["OSLR", "Stacked LSTM", "Bidirectional LSTM", "Vanilla LSTM"])

    if option == "OSLR":
        oslr_model()
    elif option == "Stacked LSTM":
        stacked_lstm_model()
    elif option == "Bidirectional LSTM":
        bidirectional_lstm_model()
    elif option == "Vanilla LSTM":
        vanilla_lstm_model()

if __name__ == "__main__":
    main()

