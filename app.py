
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px


import streamlit as st

# Custom CSS for sidebar
st.markdown("""
    <style>
    /* Sidebar width */
    [data-testid="stSidebar"] {
        min-width: 20% !important;
        max-width: 20% !important;
    }

    /* Sidebar background and text */
    [data-testid="stSidebar"] > div:first-child {
        background-color: rgba(0, 0, 0, 0.8);  /* dark theme */
        color: white;
        padding: 20px;
        border-radius: 10px;
    }

    /* Radio buttons styling */
    div[role="radiogroup"] > label {
        background-color: #2C2C2C;
        color: #FFFFFF;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 16px;
    }

    /* Hover effect */
    div[role="radiogroup"] > label:hover {
        background-color: #4CAF50; /* green highlight */
        color: white;
    }

    /* Selected option */
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #2196F3; /* blue for active */
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)




# ---------------- Page Configuration ---------------- #

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- Load Files ---------------- #

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"tesla_lstm_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load(r"scaler.pkl")

@st.cache_resource
def load_config():
    return joblib.load(r"config.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(r"Book1.csv")

model = load_model()
scaler = load_scaler()
config = load_config()
df = load_data()
y_pred_ac=pd.read_csv(r"y_pred_ac.csv",header=None).squeeze()
y_test_ac=pd.read_csv(r"y_test_ac.csv",header=None).squeeze()
val_loss=pd.read_csv(r"val_loss.csv",header=None).squeeze()
loss=pd.read_csv(r"loss.csv",header=None).squeeze()


# ---------------- Sidebar ---------------- #

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Update Data",
        "🤖 Let's Predict",
        "📈 Model Performance",
        "ℹ️ About"
    ]
)

# ---------------- HOME ---------------- #

if page == "🏠 Home":

    st.title("📈 Tesla Stock Price Prediction using LSTM")

    st.write("""
This project predicts Tesla's future closing prices using a Long Short-Term Memory (LSTM)
Deep Learning model trained on historical stock market data.
""")

    col1,col2,col3=st.columns(3)

    col1.metric("R² Score","87.91%")
    col2.metric("MAE","9.73")
    col3.metric("RMSE","12.13")

    st.success("Model Loaded Successfully ✅")
    st.info("Select 'Let's Predict' to make predictions.")
    st.info("Select 'Update Data' to update the dataset.")
    st.info("Select 'Model Performance' to view model performance.")
    st.warning("For best experience desktop :-)")

# ---------------- DATA ---------------- #

elif page == "📊 Update Data":

    st.title("Update Dataset")
    st.warning("Don't do mistake in updating data it will affect model performance.")
    st.warning("Only update data if you know what you are doing.")

    # Form for new data entry
    with st.form("update_form", clear_on_submit=True):
        new_date = st.date_input("Date")  # user picks a date
        new_open = st.number_input("Open Price", min_value=0.0, format="%.2f")
        new_close = st.number_input("Close Price", min_value=0.0, format="%.2f")
        new_high = st.number_input("High Price", min_value=0.0, format="%.2f")
        new_low = st.number_input("Low Price", min_value=0.0, format="%.2f")
        new_volume = st.number_input("Volume", min_value=0, step=1)

        submitted = st.form_submit_button("Add Row ✅")

    if submitted:
        # Convert date to string in required format (DD-MM-YYYY)
        new_date_str = pd.to_datetime(new_date).strftime("%d-%m-%Y")

        # Create new row with "Date" column (capital D)
        new_row = {
            "Date": new_date_str,
            "open": new_open,
            "close": new_close,
            "high": new_high,
            "low": new_low,
            "volume": new_volume
        }

        # Append to df
        # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # ✅ Save updated dataset back to Book1.csv
        # df.to_csv(r"C:\Users\mohak\Desktop\TSLA Stock prediction\Data\Book1.csv", index=False)

        # st.success("New data added successfully!")
        st.error("Sorry for the inconvenience, update data functionality is currently disabled. Please wait for the next update.")

        # Show updated dataset
        st.write("### Updated Dataset")
        st.dataframe(df.tail(10), use_container_width=True)
        df = load_data()

        # Updated chart
        fig = px.line(
            df,
            x="Date",
            y="close",
            title="Tesla Closing Price (Updated)"
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------- PREDICTION ---------------- #





###
elif page == "🤖 Let's Predict":
    df=load_data()

    st.title("📈 Future Tesla Stock Prediction")

    # days = st.selectbox(
    #     "Select Prediction Days",
    #     [7, 15]
    # )

    st.subheader("📅 Last Day Available Trading Data")
    # st.dataframe(df.tail(1))
    st.table(df.tail(1).style.set_table_styles(
        [{'selector': 'th', 'props': [('background-color', "#BD3D3D08"), ('color', 'white')]}]
    ))


    if st.button("Predict", use_container_width=True, type="primary", disabled=False):

        with st.spinner("Predicting Future Prices..."):

            time_steps = config["time_steps"]
            target_index = config["target_index"]

            # Last 75 days
            last_sequence = scaler.transform(
                df[config["features"]]
            )

            current_batch = last_sequence[-time_steps:].copy()

            


            pred_scaled=model.predict(
                current_batch.reshape(
                    1,
                    time_steps,
                    len(config["features"])
                ),
                verbose=0
            )[0][0]

            

            # st.success(f"Predicted Closing Price: ${pred_scaled:.2f}")
            # last date in your data
            last_date = df["Date"].iloc[-1]
            last_date = pd.to_datetime(last_date)
            # add one day
            next_date = (last_date + pd.Timedelta(days=1)).date()

            # st.write(next_date)
           
            dummy=np.zeros((1,5))
            dummy[0,3]=pred_scaled
            inverse = scaler.inverse_transform(dummy)[0][3]
            st.success(f"Predicted Closing Price: :blue[**${inverse:.2f}**] of next day i.e :blue[**{next_date}**]",icon="✅",)

            import plotly.express as px
            import plotly.graph_objects as go

            # Base chart with last 75 days
            fig = px.line(
                df.tail(75),
                x="Date",
                y="close",
                markers=True,
                title="Tesla Closing Price Prediction for Next-Day Forecast"
            )

            # Add predicted point separately
            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[inverse],
                mode="markers+text",
                marker=dict(color="red"),
                text=["Predicted"],
                textposition="top center",
                name="Prediction"
            ))
            
            

            st.plotly_chart(fig, use_container_width=True)


            
            

            
# ---------------- PERFORMANCE ---------------- #

elif page=="📈 Model Performance":

    st.title("Model Performance")

    st.metric("R²","87.91%")

    st.metric("MAE","9.73")

    st.metric("RMSE","12.13")

    import pandas as pd
    import plotly.express as px

    df_compare = pd.DataFrame({
        "Index": range(len(y_test_ac)),
        "Actual": y_test_ac,
        "Predicted": y_pred_ac
    })

    # Plot with custom colors and no markers
    fig = px.line(
        df_compare,
        x="Index",
        y=["Actual", "Predicted"],
        title="Actual vs Predicted Closing Price",
        color_discrete_map={
            "Actual": "#E63946",      # deep red
            "Predicted": "#2A9D8F"   # teal green
        }
    )

    # Detailed styling
    fig.update_layout(
        xaxis=dict(
            title="Index",
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="black"
        ),
        yaxis=dict(
            title="Closing Price",
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="black"
        ),
        legend=dict(
            title="Legend",
            bgcolor="rgba(240,240,240,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        font=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("This graph compares the actual Tesla stock closing prices with the prices predicted by our LSTM model on the test dataset. The predicted curve closely follows the trend of the actual stock prices, indicating that the model has successfully learned the underlying patterns and temporal dependencies in the historical data. Although there are small differences at certain points, which are expected due to market volatility, the predicted values remain very close to the actual values. This demonstrates that the model generalizes well to unseen data and is capable of making reliable short-term stock price predictions.")


    import pandas as pd
    import plotly.express as px

    # Build DataFrame
    loss_df = pd.DataFrame({
        "Epoch": range(1, len(loss)+1),
        "Training Loss": loss,
        "Validation Loss": val_loss
    })

    # Plot both lines with custom colors
    fig = px.line(
        loss_df,
        x="Epoch",
        y=["Training Loss", "Validation Loss"],
        title="Training vs Validation Loss",
        color_discrete_map={
            "Training Loss": "#E63946",   # deep red
            "Validation Loss": "#2A9D8F"  # teal green
        }
    )

    # Detailed styling
    fig.update_layout(
        xaxis=dict(
            title="Epoch",
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="black"
        ),
        yaxis=dict(
            title="Loss",
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="black"
        ),
        legend=dict(
            title="Legend",
            bgcolor="rgba(240,240,240,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        font=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("The loss and validation loss curves decrease together and remain very close throughout training. Since there is no large gap between them, the model is not overfitting. Also, both losses reach low values, indicating that the model is not underfitting. This means the model has learned the data well and can generalize effectively to new, unseen stock prices")






# ---------------- ABOUT ---------------- #
elif page=="ℹ️ About":
    st.title("📖 About This Project")

    st.subheader("🤖 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Algorithm:** LSTM

        **Framework:** TensorFlow/Keras

        **Deployment:** Streamlit
        """)

    with col2:
        st.info("""
        **Dataset:** Tesla Historical Stock Data

        **Features:** Open, High, Low, Close, Volume

        **Window Size:** 75 Days
        """)

    st.subheader("📈 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", "87.91%")
    c2.metric("MAE", "9.73")
    c3.metric("RMSE", "12.13")

    with st.expander("⚙ Hyperparameter Tuning", expanded=True):

        st.write("""
        ✔ LSTM(100)

        ✔ Dropout(0.20)

        ✔ LSTM(50)

        ✔ Dropout(0.20)

        ✔ Dense(25)

        ✔ Dense(1)

        ✔ Optimizer : Adam

        ✔ Loss : Huber Loss

        ✔ ReduceLROnPlateau

        ✔ EarlyStopping

        ✔ Learning Rate : 0.01

        ✔ Batch Size : 20

        ✔ Time Steps : 75

        ✔ MinMaxScaler
        """)

    st.success("""

    ✅ Used Huber Loss for robust learning

    ✅ ReduceLROnPlateau for automatic LR tuning

    ✅ EarlyStopping to prevent overfitting

    ✅ MinMaxScaler for feature normalization

    ✅ Dropout to reduce overfitting

    ✅ Five market features used
    """)

    st.subheader("🎯 Project Outcome")

    st.success("""

    ✔ Stable Training & Validation Loss

    ✔ No Significant Overfitting

    ✔ Accurate Trend Prediction

    ✔ Reliable Next-Day Stock Price Prediction
    """)
