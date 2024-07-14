import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set page configuration
st.set_page_config(page_title="Stock Prediction App", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for background image and text styles
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .reportview-container {
        background: url("https://example.com/background-image.jpg");
        background-size: cover;
    }
    .title {
        color: #2c3e50;
        font-size: 50px;
        font-weight: bold;
        width: 100%;
        text-align: center;
        margin-top: 20px;
    }
    .header {
        color: #34495e;
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        color: #7f8c8d;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and headers
st.markdown('<div class="title">Stock Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Họ và tên: Nguyễn Hữu Thiện</div>', unsafe_allow_html=True)
st.markdown('<div class="header">MSSV: 20120194</div>', unsafe_allow_html=True)

# Spacer and introduction
st.write("")
st.markdown('<div class="subheader">Bài tập cá nhân Stock prediction</div>', unsafe_allow_html=True)
st.write("Welcome to the Stock Prediction App. Use the sidebar to select options and the main area to view results.")
st.write("")

# User inputs
stocks = ('BTC-USD', 'ETH-USD', 'ADA-USD')
selected_stock = st.selectbox('Chọn cặp tiền dự đoán', stocks)
start_date = st.date_input('Chọn ngày bắt đầu', value=date(2018, 1, 1))
TODAY = date.today().strftime("%Y-%m-%d")

# Load data function
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

# Load and display data
data_load_state = st.text('Đang tải...')
data = load_data(selected_stock, start_date, TODAY)
data_load_state.text('Đã tải dữ liệu xong!')
st.write(data.head())
st.write(data.tail())

# Prepare data for Prophet model
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Prediction
n_years = st.slider('Số năm dự đoán', 1, 4)
period = n_years * 365

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Dự báo dữ liệu')

@st.cache_data(ttl=24*60*60)  # cache for 24 hours
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(data)

st.download_button(
    label="Tải về dữ liệu",
    data=csv,
    file_name=f"{selected_stock}.csv",
    mime="text/csv"
)
st.write(forecast)

# Display forecast plot
st.subheader(f'Dự báo trong {n_years} năm')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Plot raw data function
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Giá mở"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Giá đóng"))
    fig.layout.update(title_text='Biểu diễn biểu đồ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Display forecast components
st.subheader("Các thành phần")
fig2 = m.plot_components(forecast)
st.write(fig2)
