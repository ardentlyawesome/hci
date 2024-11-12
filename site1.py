import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
from streamlit_option_menu import option_menu

# Custom CSS to improve styling
# Dark mode CSS styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
        font-family: Arial, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffcc00;  /* Gold color for headers */
        margin-bottom: 16px;
    }
    .stApp {
        padding-top: 30px;  /* Adequate padding to space the content */
    }
    .css-1d391kg { /* Adjust font size of sidebar menu */
        font-size: 18px;
    }
    .css-15tx938 { /* Customize the sidebar width */
        width: 260px;
    }
    .css-17eq0hr a {  /* Link customization */
        color: #ffcc00 !important;  /* Links in the sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Navbar with buttons for About and Contact Us
# st.markdown(
#     """
#     <div class="navbar">
#         <button onclick="window.location.href = '#about';">About</button>
#         <button onclick="window.location.href = '#contact';">Contact Us</button>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# Function to load and preprocess data (for Vanilla LSTM, for example)
def load_and_preprocess_data():
    column_names = ["Date", "OilPrice"]
    brent_data = pd.read_excel(
        "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls",
        sheet_name=1,
        skiprows=4,
        engine='xlrd',
        names=column_names
    )
    brent_data.dropna(inplace=True)  # Drop missing values if any
    brent_data['Date'] = pd.to_datetime(brent_data['Date'])
    brent_data.set_index('Date', inplace=True)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    brent_data['normalized_oil_prices'] = scaler.fit_transform(brent_data[['OilPrice']])
    
    return brent_data, scaler

# Function to load Vanilla LSTM model
def load_lstm_model():
    model = load_model('vanilla.h5')  # Update path as needed
    return model

# Load Stacked LSTM Model
def load_stacked_lstm_model():
    model = load_model('stacked_lstm_oil_price_model.h5')  # Update with the actual path to your Stacked LSTM model
    return model

# Load Bidirectional LSTM Model
def load_bidirectional_lstm_model():
    model = load_model('bi_di_LSTM.h5')  # Update with the actual path to your Bidirectional LSTM model
    return model

def load_oslr_model():
    model = load_model('oslr.h5')  # Update with the actual path to your Bidirectional LSTM model
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

# Homepage Function
def home_page():
    st.title("📈 Oil Price Prediction Dashboard")
    st.write("""
        Welcome to our LSTM Model Dashboard! This platform showcases different variations of Long Short-Term Memory (LSTM) models, including Vanilla LSTM, Bidirectional LSTM, our custom Online Stream LSTM Regression, and Stacked LSTM. 
        Each model is designed to handle sequential data in unique ways, offering insights into different aspects of time-series prediction.
        You can explore interactive plots for each model, comparing performance across different date ranges to suit your analysis needs. 
        Simply select a model and adjust the date range to visualize its predictions in real time.
    """)
    
    # Load and preprocess the data
    brent_data, scaler = load_and_preprocess_data()
    
    # Get the latest available oil price
    latest_date = brent_data.index.max()
    latest_price = brent_data.loc[latest_date, 'OilPrice']
    
    st.subheader("💲 Current Oil Price")
    st.write(f"As of **{latest_date.date()}**, the oil price is **${latest_price:.2f}** per barrel.")
    
    # Display recent oil prices
    st.subheader("📉 Recent Oil Prices")
    recent_data = brent_data.tail(60)  # Last 60 entries (approx. 5 years if monthly data)
    st.line_chart(recent_data['OilPrice'])

# About Page
def about_page():
    st.title("📝 About Us")
    st.write("""
        This project is a comprehensive application developed to predict oil prices using various machine learning models.
        We aim to provide accurate predictions and insights for oil price movements using state-of-the-art techniques like LSTM and Bidirectional LSTM.
    """)

# Contact Us Page
def contact_page():
    st.title("📞 Contact Us")
    st.write("""
        For any inquiries or suggestions, feel free to reach out to us via the following channels:
        
        - **Email:** support@oilpricepredictions.com
        - **Phone:** +123-456-7890
        - **Address:** 123 Machine Learning Street, Data City, AI Country
    """)

# Page 1: OSLR Model
def oslr_model():
    st.title("OSLR Model")
    st.write("""Our methodology focuses on improving crude oil price forecasting by leveraging Long Short-Term Memory (LSTM) networks combined with online learning. We use the Brent Crude Oil Price dataset (2010-2023) and split it into training and validation sets. 
        Our core model is a standard LSTM that predicts future prices by analyzing past trends through a sliding window approach. To enhance this, we developed the Online Sequential LSTM Regression (OSLR) model, which updates continuously with new data, allowing it to capture evolving market patterns in real time. 
        The OSLR model is evaluated using key metrics like Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Directional Accuracy (DA), and consistently outperforms traditional LSTM models in terms of accuracy and error reduction.
            """)
    


     # Load and preprocess the data
    brent_data, scaler = load_and_preprocess_data()
    
    # Load 
    model = load_oslr_model()
    
    # Make predictions
    predictions = make_predictions(model, brent_data['normalized_oil_prices'])
    
    # Date range input
    st.write("### Select Date Range:")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    
    # Validate date inputs
    if start_date > end_date:
        st.error("Error: Start Date must be before End Date.")
        return
    
    # Filter data based on date range
    filtered_test_data = brent_data.loc[start_date:end_date]['normalized_oil_prices']
    filtered_predictions = predictions[brent_data.index.get_indexer(filtered_test_data.index)]
    
    # Plot the predictions vs actual values within the selected date range
    plot_predictions(filtered_test_data, filtered_predictions, start_date, end_date)

# Page 2: Stacked LSTM Model
def stacked_lstm_model():
    st.title("Stacked LSTM Model")
    st.write("""A Stacked LSTM is a type of LSTM model that consists of multiple layers of LSTM cells stacked on top of each other. 
        By adding more layers, the model can capture more complex patterns in the data, allowing it to learn hierarchical representations of the input sequence. 
        This architecture is particularly useful for tasks requiring deep feature extraction, such as time-series forecasting or speech recognition, as it enables the model to process information at multiple levels of abstraction.
    """)
    
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
    st.write("""
        A Bidirectional Long Short-Term Memory (BiLSTM) network extends the vanilla LSTM by processing data in both forward and backward directions. 
        It uses two separate LSTMs: one for the input sequence and another for the reversed sequence. 
        This allows the network to capture dependencies from both past and future context, improving performance in tasks like language modeling, speech recognition, and machine translation where full context is important.
        """)
    
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


# Page 4: Vanilla LSTM Model
def vanilla_lstm_model():
    st.title("☁️ Vanilla LSTM Model")
    st.write("""
        A vanilla Long Short-Term Memory (LSTM) network is a type of recurrent neural network (RNN) that captures long-term dependencies in sequential data, solving the vanishing gradient problem of traditional RNNs. 
        It uses a memory cell and three gates—forget, input, and output—to control the flow of information over time. 
        This allows LSTMs to retain important data over long sequences, making them useful for tasks like time-series forecasting and natural language processing.
    """)
    
    # Load and preprocess the data
    brent_data, scaler = load_and_preprocess_data()
    
    # Load the Vanilla LSTM model
    model = load_lstm_model()
    
    # Make predictions
    predictions = make_predictions(model, brent_data['normalized_oil_prices'])
    
    # Date range input
    st.write("### Select Date Range:")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    
    # Validate date inputs
    if start_date > end_date:
        st.error("Error: Start Date must be before End Date.")
        return
    
    # Filter data based on date range
    filtered_test_data = brent_data.loc[start_date:end_date]['normalized_oil_prices']
    filtered_predictions = predictions[brent_data.index.get_indexer(filtered_test_data.index)]
    
    # Plot the predictions vs actual values within the selected date range
    plot_predictions(filtered_test_data, filtered_predictions, start_date, end_date)

# Function to display Results page with model performance comparison table
def results_page():
    st.title("📊 Model Results")
    
    # Brief description of the metrics
    st.subheader("🔍 Metrics Explanation")
    st.write("""
        - *RMSE (Root Mean Squared Error)*: Measures the average magnitude of prediction errors by calculating the square root of the average squared differences between predicted and actual values. It is sensitive to large errors.
        - *MAE (Mean Absolute Error)*: Computes the average of absolute differences between predicted and actual values, providing a straightforward measure of prediction accuracy without emphasizing larger errors.
        - *Directional Accuracy*: Evaluates the model's ability to correctly predict the direction of change (up or down) in the data series, focusing on trend prediction accuracy rather than exact values.
    """)
    
    # Table for displaying RMSE, MAE, and Directional Accuracy for different models
    metrics_data = {
        "Model": ["Vanilla LSTM", "Stacked LSTM", "Bidirectional LSTM", "OSLR"],
        "RMSE": [0.234, 0.210, 0.221, 0.185],  # Example RMSE values
        "MAE": [0.176, 0.164, 0.172, 0.149],   # Example MAE values
        "Directional Accuracy": [0.92, 0.94, 0.93, 0.95]  # Example directional accuracy values
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display the table
    st.write("### Model Performance Metrics")
    st.dataframe(metrics_df)



# Main function
def main():
    # Sidebar for navigation
    with st.sidebar:
        st.title("🔍 Navigation")

        # Combined navigation menu with custom styling
        selected = option_menu(
            menu_title="Menu",
            options=["Home", "OSLR", "Stacked LSTM", "Bidirectional LSTM", "Vanilla LSTM", "Results", "About Us", "Contact Us"],
            icons=["house", "bar-chart", "graph-up", "shuffle", "cloud", "clipboard-data", "info-circle", "envelope"],
            menu_icon="cast",
            default_index=0,
        )

    # Render only the selected page
    if selected == "Home":
        home_page()
        st.success("You are now viewing the Home page.")
    elif selected == "About Us":
        about_page()
        st.success("You are now viewing the About Us page.")
    elif selected == "Contact Us":
        contact_page()
        st.success("You are now viewing the Contact Us page.")
    elif selected == "OSLR":
        oslr_model()
        st.success("You are now viewing the OSLR model.")
    elif selected == "Stacked LSTM":
        stacked_lstm_model()
        st.success("You are now viewing the Stacked LSTM model.")
    elif selected == "Bidirectional LSTM":
        bidirectional_lstm_model()
        st.success("You are now viewing the Bidirectional LSTM model.")
    elif selected == "Vanilla LSTM":
        vanilla_lstm_model()
        st.success("You are now viewing the Vanilla LSTM model.")
    elif selected == "Results":
        results_page()
        st.success("You are now viewing the Results page.")



if __name__ == "__main__":
    main()
