import streamlit as st
import os
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json
import uuid
import datetime
import yfinance as yf
import wbgapi as wb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from gtts import gTTS
import base64

# Set page config as the first Streamlit command
st.set_page_config(page_title="Financial Q&A Assistant", layout="wide")

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
    
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# Load environment variables
load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")
GOLDAP_API_KEY = os.getenv("GOLDAP_API")
FINNHUB_API_KEY = os.getenv("FINNHUB_API")
NINJAS_API_KEY = os.getenv("NINJAS_API")
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Load local CSV data
csv_path = 'financial_risk_analysis_large.csv'
try:
    financial_data = pd.read_csv(csv_path)
except FileNotFoundError:
    financial_data = pd.DataFrame()
    st.warning("Note: financial_risk_analysis_large.csv not found. Using alternative data sources.")

# Expanded list of companies
COMPANY_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
    "GOOGL", "META", "BRK-B", "AVGO", "TSM",
    "TM", "BABA", "V", "WMT", "JPM"
]

# Enhanced data fetching with yfinance
@st.cache_data
def get_yfinance_data(symbols=COMPANY_SYMBOLS, period="6mo"):
    """Fetch stock data using yfinance"""
    all_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            
            if not hist.empty:
                all_data[symbol] = {
                    'history': hist,
                    'info': info
                }
        except Exception as e:
            st.warning(f"Could not fetch yfinance data for {symbol}: {e}")
    return all_data

# World Bank data fetching
@st.cache_data
def get_world_bank_data():
    """Fetch economic indicators from World Bank API"""
    try:
        countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'BRA']
        gdp_data = wb.data.DataFrame('NY.GDP.MKTP.CD', countries, time=range(2018, 2023))
        inflation_data = wb.data.DataFrame('FP.CPI.TOTL.ZG', countries, time=range(2018, 2023))
        unemployment_data = wb.data.DataFrame('SL.UEM.TOTL.ZS', countries, time=range(2018, 2023))
        
        return {
            'gdp': gdp_data,
            'inflation': inflation_data,
            'unemployment': unemployment_data
        }
    except Exception as e:
        st.warning(f"Could not fetch World Bank data: {e}")
        return {}

# Enhanced stock data with fallback
@st.cache_data
def get_stock_data(symbols=COMPANY_SYMBOLS):
    """Fetch stock data with Alpha Vantage fallback"""
    all_data = {}
    for symbol in symbols:
        if ALPHA_API_KEY:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_API_KEY}&outputsize=compact"
            try:
                response = requests.get(url)
                data = response.json()
                if 'Time Series (Daily)' in data:
                    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float)
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)
                    all_data[symbol] = df
                    continue
            except Exception as e:
                st.warning(f"Alpha Vantage failed for {symbol}: {e}")
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            if not hist.empty:
                all_data[symbol] = hist
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {e}")
    
    return all_data

@st.cache_data
def get_company_profile(symbols=COMPANY_SYMBOLS):
    """Enhanced company profile with yfinance fallback"""
    all_data = {}
    for symbol in symbols:
        if FMP_API_KEY:
            url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FMP_API_KEY}"
            try:
                response = requests.get(url)
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    all_data[symbol] = data[0]
                    continue
            except Exception as e:
                st.warning(f"FMP failed for {symbol}: {e}")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info:
                all_data[symbol] = {
                    'symbol': symbol,
                    'companyName': info.get('longName', symbol),
                    'mktCap': info.get('marketCap', 0),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'price': info.get('currentPrice', 0)
                }
        except Exception as e:
            st.warning(f"Could not fetch profile for {symbol}: {e}")
    
    return all_data

@st.cache_data
def get_exchange_rate(base="USD", symbols="EUR,JPY,GBP,CAD,AUD,CHF,CNH,HKD,NZD,KWD"):
    """Fetch exchange rates"""
    if not OER_API_KEY:
        return {
            'rates': {
                'EUR': 0.85,
                'JPY': 110.0,
                'GBP': 0.75,
                'CAD': 1.25,
                'AUD': 1.35,
                'CHF': 0.92
            }
        }
    
    url = f"https://openexchangerates.org/api/latest.json?app_id={OER_API_KEY}&base={base}&symbols={symbols}"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        st.warning(f"Error fetching exchange rate data: {e}")
        return {'rates': {}}

@st.cache_data
def get_gold_price():
    """Fetch gold price with fallback"""
    if not GOLDAP_API_KEY:
        return {'price': 2000.0, 'currency': 'USD'}
    
    symbol = "XAU"
    curr = "USD"
    url = f"https://www.goldapi.io/api/{symbol}/{curr}"
    headers = {
        "x-access-token": GOLDAP_API_KEY,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching gold price data: {e}")
        return {'price': 2000.0, 'currency': 'USD'}

# Fetch all data
yfinance_data = get_yfinance_data()
world_bank_data = get_world_bank_data()
alpha_data = get_stock_data()
fmp_data = get_company_profile()
exchange_rate_data = get_exchange_rate()
gold_price_data = get_gold_price()

# Convert all data into LangChain documents
def convert_data_to_documents(combined_data):
    documents = []
    
    if "csv_data" in combined_data and combined_data["csv_data"]:
        for row in combined_data["csv_data"][:100]:
            content = "\n".join([f"{k}: {v}" for k, v in row.items() if v is not None])
            documents.append(Document(page_content=content, metadata={"source": "csv"}))
    
    if "yfinance_data" in combined_data and combined_data["yfinance_data"]:
        for symbol, data in combined_data["yfinance_data"].items():
            if 'info' in data:
                content = f"Company: {symbol}\n" + "\n".join([f"{k}: {v}" for k, v in data['info'].items() if v is not None])
                documents.append(Document(page_content=content, metadata={"source": "yfinance", "symbol": symbol}))
    
    if "world_bank_data" in combined_data and combined_data["world_bank_data"]:
        for indicator, data in combined_data["world_bank_data"].items():
            if hasattr(data, 'to_string'):
                content = f"World Bank {indicator} data:\n{data.to_string()}"
                documents.append(Document(page_content=content, metadata={"source": "world_bank", "indicator": indicator}))
    
    for key, data in combined_data.items():
        if key not in ["csv_data", "yfinance_data", "world_bank_data"] and data:
            documents.append(Document(page_content=str(data), metadata={"source": key}))
    
    return documents

# Combine all data
combined_data = {
    "csv_data": financial_data.to_dict(orient="records") if not financial_data.empty else [],
    "yfinance_data": yfinance_data,
    "world_bank_data": world_bank_data,
    "stock_data_alpha": alpha_data,
    "company_profile_fmp": fmp_data,
    "exchange_rate": exchange_rate_data,
    "gold_price": gold_price_data
}

# Convert to documents
documents = convert_data_to_documents(combined_data)

# Initialize embeddings and vectorstore
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    
    if os.path.exists("faiss_index") and OPENAI_API_KEY:
        try:
            vectorstore = FAISS.load_local("faiss_index", embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
        except:
            if OPENAI_API_KEY:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                vectorstore.save_local("faiss_index")
    else:
        if OPENAI_API_KEY:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local("faiss_index")

# Define prompt template
prompt_template = """
You are a financial Q&A assistant with access to comprehensive financial data including stock prices, company profiles, economic indicators, exchange rates, and World Bank economic data. Use the following context to answer questions accurately and provide insights.

Context:
{context}

Question:
{question}

Provide a comprehensive answer using the available data. If you need to make calculations or comparisons, show your work. Respond in the same language as the question.

Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize LLM and RAG chain
if OPENAI_API_KEY and 'vectorstore' in locals():
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Main UI
st.title("🔍 Enhanced Financial Q&A Assistant")
st.markdown("Ask about stocks, economic indicators, company profiles, or global economic data. Now powered by Yahoo Finance and World Bank data!")

# Display data sources status
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Yahoo Finance", len(yfinance_data), "Companies")
with col2:
    st.metric("World Bank", len(world_bank_data), "Indicators")
with col3:
    st.metric("Stock Data", len(alpha_data), "Symbols")
with col4:
    st.metric("Company Profiles", len(fmp_data), "Profiles")

# Chat interface
if OPENAI_API_KEY:
    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input("Your question:", placeholder="e.g., Compare Tesla's performance with Apple's market cap", key="question_input")
        submit_button = st.form_submit_button("Ask")

    if question and submit_button and 'rag_chain' in locals():
        with st.spinner("Analyzing financial data..."):
            try:
                response = rag_chain.invoke({"query": question})
                answer = response["result"]
                sources = response["source_documents"]

                st.session_state.conversation_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })

                st.subheader("💡 Answer")
                st.write(answer)
                
                if sources:
                    with st.expander("📊 Data Sources Used"):
                        for i, doc in enumerate(sources, 1):
                            st.write(f"**Source {i} ({doc.metadata.get('source', 'unknown')})**:")
                            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

                try:
                    tts = gTTS(text=answer, lang='ar' if any(c in question for c in 'ابتثجحخدذرزسشصضطظعغفقكلمنهوية') else 'en')
                    tts.save("output.mp3")
                    audio_file = open("output.mp3", "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mpeg")
                    st.download_button("Download Audio", audio_bytes, "output.mp3", "audio/mpeg")
                except Exception as e:
                    st.warning(f"Could not generate audio: {e}")

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
else:
    st.warning("Please set your OPENAI_API_KEY in the .env file to use the Q&A feature.")

# Enhanced Data Visualization Section
st.header("📈 Interactive Data Visualization")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stock Analysis", "Company Comparison", "Economic Indicators", "Exchange Rates", "Gold & Commodities"])

with tab1:
    st.subheader("📊 Stock Price Analysis")
    
    selected_stocks = st.multiselect("Select stocks to analyze:", COMPANY_SYMBOLS, default=["AAPL", "MSFT", "TSLA"])
    
    if selected_stocks:
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=('Stock Prices', 'Trading Volume'),
                          vertical_spacing=0.1)
        
        colors = px.colors.qualitative.Set1
        
        for i, symbol in enumerate(selected_stocks):
            color = colors[i % len(colors)]
            
            if symbol in yfinance_data and 'history' in yfinance_data[symbol]:
                df = yfinance_data[symbol]['history']
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['Close'], name=f"{symbol} Price", 
                             line=dict(color=color, width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Bar(x=df.index, y=df['Volume'], name=f"{symbol} Volume", 
                          marker_color=color, opacity=0.6),
                    row=2, col=1
                )
            elif symbol in alpha_data:
                df = alpha_data[symbol]
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['4. close'], name=f"{symbol} Price", 
                             line=dict(color=color, width=2)),
                    row=1, col=1
                )
                if '5. volume' in df.columns:
                    fig.add_trace(
                        go.Bar(x=df.index, y=df['5. volume'], name=f"{symbol} Volume", 
                              marker_color=color, opacity=0.6),
                        row=2, col=1
                    )
        
        fig.update_layout(height=600, title_text="Stock Analysis Dashboard")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🏢 Company Market Cap Comparison")
    
    if fmp_data:
        market_caps = []
        companies = []
        sectors = []
        
        for symbol, data in fmp_data.items():
            if isinstance(data, dict) and data.get('mktCap'):
                market_caps.append(data['mktCap'] / 1e9)  # Convert to billions
                companies.append(symbol)
                sectors.append(data.get('sector', 'Unknown'))
        
        if market_caps:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=companies,
                y=market_caps,
                text=[f"${cap:.1f}B" for cap in market_caps],
                textposition='auto',
                marker_color=px.colors.qualitative.Set3
            ))
            
            fig.update_layout(
                title="Market Capitalization Comparison",
                xaxis_title="Company",
                yaxis_title="Market Cap (Billions USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Largest Company", companies[market_caps.index(max(market_caps))], f"${max(market_caps):.1f}B")
            with col2:
                st.metric("Smallest Company", companies[market_caps.index(min(market_caps))], f"${min(market_caps):.1f}B")
            with col3:
                st.metric("Average Market Cap", f"${np.mean(market_caps):.1f}B")

with tab3:
    st.subheader("🌍 Economic Indicators (World Bank Data)")
    
    if world_bank_data:
        indicator = st.selectbox("Select Economic Indicator:", 
                                ["GDP", "Inflation", "Unemployment"], 
                                key="wb_indicator")
        
        if indicator.lower() in world_bank_data:
            data = world_bank_data[indicator.lower()]
            
            if hasattr(data, 'reset_index'):
                df = data.reset_index()
                
                fig = go.Figure()
                
                for country in df.columns[1:]:
                    if country in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.iloc[:, 0],
                            y=df[country],
                            mode='lines+markers',
                            name=country,
                            line=dict(width=2)
                        ))
                
                fig.update_layout(
                    title=f"{indicator} Trends by Country",
                    xaxis_title="Year",
                    yaxis_title=f"{indicator} Value",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("World Bank data not available. Please check your internet connection.")

with tab4:
    st.subheader("💱 Exchange Rates (USD Base)")
    
    if exchange_rate_data and 'rates' in exchange_rate_data:
        rates = exchange_rate_data['rates']
        
        if rates:
            currencies = list(rates.keys())
            values = list(rates.values())
            
            fig = go.Figure(go.Bar(
                x=values,
                y=currencies,
                orientation='h',
                marker_color=px.colors.sequential.Viridis,
                text=[f"{rate:.3f}" for rate in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Current Exchange Rates (1 USD = X Currency)",
                xaxis_title="Exchange Rate",
                yaxis_title="Currency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No exchange rate data available.")

with tab5:
    st.subheader("🥇 Gold Price & Commodities")
    
    if gold_price_data and 'price' in gold_price_data:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gold Price", f"${gold_price_data['price']:.2f}", 
                     f"{gold_price_data.get('currency', 'USD')} per ounce")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = gold_price_data['price'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Gold Price (USD/oz)"},
            gauge = {
                'axis': {'range': [None, 2500]},
                'bar': {'color': "gold"},
                'steps': [
                    {'range': [0, 1500], 'color': "lightgray"},
                    {'range': [1500, 2000], 'color': "yellow"},
                    {'range': [2000, 2500], 'color': "orange"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2200}
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Gold price data not available.")

# Sidebar for conversation history
with st.sidebar:
    st.header("💬 Conversation History")
    if st.session_state.conversation_history:
        for i, entry in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q{i+1}: {entry['question'][:50]}..."):
                st.write(f"**Question**: {entry['question']}")
                st.write(f"**Answer**: {entry['answer'][:200]}...")
                if entry['sources']:
                    st.write(f"**Sources**: {', '.join([doc.metadata.get('source', 'unknown') for doc in entry['sources']])}")
    
    if st.button("🔄 New Conversation"):
        st.session_state.conversation_history = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()
    
    st.header("📊 Data Sources")
    st.write("✅ Yahoo Finance")
    st.write("✅ World Bank API")
    st.write("✅ Alpha Vantage" if ALPHA_API_KEY else "⚠️ Alpha Vantage (API key needed)")
    st.write("✅ Financial Modeling Prep" if FMP_API_KEY else "⚠️ FMP (API key needed)")
    st.write("✅ Exchange Rates" if OER_API_KEY else "⚠️ Exchange Rates (mock data)")
    st.write("✅ Gold API" if GOLDAP_API_KEY else "⚠️ Gold API (mock data)")

# Footer
st.markdown("---")
st.markdown("**Enhanced Financial Q&A Assistant** - Powered by AI, Yahoo Finance, and World Bank Data")