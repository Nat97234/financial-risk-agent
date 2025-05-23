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
from flask import Flask, jsonify, render_template, request
import random
import sys
import argparse

# تحميل متغيرات البيئة
load_dotenv()

# مفاتيح API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")
GOLDAP_API_KEY = os.getenv("GOLDAP_API")
FINNHUB_API_KEY = os.getenv("FINNHUB_API")
NINJAS_API_KEY = os.getenv("NINJAS_API")
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# النصائح المالية
tips = [
    "استثمر في الشركات ذات النمو المستدام.",
    "احرص على تنويع المحفظة لتقليل المخاطر.",
    "راقب مؤشرات السوق والمؤشرات الاقتصادية العالمية."
]

# رموز الشركات
COMPANY_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
    "GOOGL", "META", "BRK-B", "AVGO", "TSM",
    "TM", "BABA", "V", "WMT", "JPM"
]

# دوال جلب البيانات
@st.cache_data
def get_yfinance_data(symbols=COMPANY_SYMBOLS, period="6mo"):
    """جلب بيانات الأسهم باستخدام yfinance"""
    all_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            if not hist.empty:
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                all_data[symbol] = {'history': hist, 'info': info}
        except Exception as e:
            print(f"فشل جلب بيانات yfinance لـ {symbol}: {e}")
    return all_data

@st.cache_data
def get_world_bank_data():
    """جلب المؤشرات الاقتصادية من البنك الدولي"""
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
        print(f"فشل جلب بيانات البنك الدولي: {e}")
        return {}

@st.cache_data
def get_stock_data(symbols=COMPANY_SYMBOLS):
    """جلب بيانات الأسهم مع الاحتياط باستخدام Alpha Vantage"""
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
                    df['MA20'] = df['4. close'].rolling(window=20).mean()
                    all_data[symbol] = df
                    continue
            except Exception as e:
                print(f"فشل Alpha Vantage لـ {symbol}: {e}")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            if not hist.empty:
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                all_data[symbol] = hist
        except Exception as e:
            print(f"فشل جلب بيانات {symbol}: {e}")
    return all_data

@st.cache_data
def get_company_profile(symbols=COMPANY_SYMBOLS):
    """جلب ملفات تعريف الشركات مع الاحتياط باستخدام yfinance"""
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
                print(f"فشل FMP لـ {symbol}: {e}")
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
            print(f"فشل جلب ملف تعريف {symbol}: {e}")
    return all_data

@st.cache_data
def get_exchange_rate(base="USD", symbols="EUR,JPY,GBP,CAD,AUD,CHF,CNH,HKD,NZD,KWD"):
    """جلب أسعار الصرف"""
    if not OER_API_KEY:
        return {'rates': {'EUR': 0.85, 'JPY': 110.0, 'GBP': 0.75, 'CAD': 1.25, 'AUD': 1.35, 'CHF': 0.92}}
    url = f"https://openexchangerates.org/api/latest.json?app_id={OER_API_KEY}&base={base}&symbols={symbols}"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"خطأ في جلب بيانات أسعار الصرف: {e}")
        return {'rates': {}}

@st.cache_data
def get_gold_price():
    """جلب سعر الذهب مع الاحتياط"""
    if not GOLDAP_API_KEY:
        return {'price': 2000.0, 'currency': 'USD'}
    symbol = "XAU"
    curr = "USD"
    url = f"https://www.goldapi.io/api/{symbol}/{curr}"
    headers = {"x-access-token": GOLDAP_API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"خطأ في جلب بيانات سعر الذهب: {e}")
        return {'price': 2000.0, 'currency': 'USD'}

# تحميل بيانات CSV
csv_path = 'financial_risk_analysis_large.csv'
try:
    financial_data = pd.read_csv(csv_path)
except FileNotFoundError:
    financial_data = pd.DataFrame()

# تحويل البيانات إلى مستندات LangChain
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

# تطبيق Streamlit
def run_streamlit():
    st.set_page_config(page_title="مساعد الأسئلة المالية", layout="wide")
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())

    # جلب البيانات
    yfinance_data = get_yfinance_data()
    world_bank_data = get_world_bank_data()
    alpha_data = get_stock_data()
    fmp_data = get_company_profile()
    exchange_rate_data = get_exchange_rate()
    gold_price_data = get_gold_price()

    combined_data = {
        "csv_data": financial_data.to_dict(orient="records") if not financial_data.empty else [],
        "yfinance_data": yfinance_data,
        "world_bank_data": world_bank_data,
        "stock_data_alpha": alpha_data,
        "company_profile_fmp": fmp_data,
        "exchange_rate": exchange_rate_data,
        "gold_price": gold_price_data
    }

    documents = convert_data_to_documents(combined_data)

    # تهيئة التضمينات ومخزن المتجهات
    if documents and OPENAI_API_KEY:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        if os.path.exists("faiss_index"):
            try:
                vectorstore = FAISS.load_local("faiss_index", embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
            except:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vectorstore = FAISS.from_documents(split_docs, embeddings)
                vectorstore.save_local("faiss_index")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local("faiss_index")

        prompt_template = """
        أنت مساعد أسئلة مالية مع إمكانية الوصول إلى بيانات مالية شاملة تشمل أسعار الأسهم، ملفات تعريف الشركات، المؤشرات الاقتصادية، أسعار الصرف، وبيانات البنك الدولي. استخدم السياق التالي للإجابة على الأسئلة بدقة وتقديم رؤى.

        السياق:
        {context}

        السؤال:
        {question}

        قدم إجابة شاملة باستخدام البيانات المتاحة. إذا كنت بحاجة إلى إجراء حسابات أو مقارنات، أظهر عملك. أجب بنفس لغة السؤال.

        الإجابة:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    # واجهة Streamlit
    st.title("🔍 مساعد الأسئلة المالية المحسن")
    st.markdown("اسأل عن الأسهم، المؤشرات الاقتصادية، ملفات الشركات، أو البيانات الاقتصادية العالمية. مدعوم الآن بـ Yahoo Finance وبيانات البنك الدولي!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Yahoo Finance", len(yfinance_data), "شركات")
    with col2:
        st.metric("البنك الدولي", len(world_bank_data), "مؤشرات")
    with col3:
        st.metric("بيانات الأسهم", len(alpha_data), "رموز")
    with col4:
        st.metric("ملفات الشركات", len(fmp_data), "ملفات")

    if OPENAI_API_KEY:
        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_input("سؤالك:", placeholder="مثال: قارن أداء تسلا مع القيمة السوقية لآبل", key="question_input")
            submit_button = st.form_submit_button("اسأل")

        if question and submit_button and 'rag_chain' in locals():
            with st.spinner("جارٍ تحليل البيانات المالية..."):
                try:
                    response = rag_chain.invoke({"query": question})
                    answer = response["result"]
                    sources = response["source_documents"]
                    st.session_state.conversation_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    })
                    st.subheader("💡 الإجابة")
                    st.write(answer)
                    if sources:
                        with st.expander("📊 المصادر المستخدمة"):
                            for i, doc in enumerate(sources, 1):
                                st.write(f"**المصدر {i} ({doc.metadata.get('source', 'غير معروف')})**:")
                                st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                except Exception as e:
                    st.error(f"خطأ في معالجة السؤال: {str(e)}")
    else:
        st.warning("يرجى تعيين OPENAI_API_KEY في ملف .env لاستخدام ميزة الأسئلة والأجوبة.")

    # قسم التصور البياني
    st.header("📈 التصور البياني التفاعلي")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["تحليل الأسهم", "مقارنة الشركات", "المؤشرات الاقتصادية", "أسعار الصرف", "الذهب والسلع"])

    with tab1:
        st.subheader("📊 تحليل أسعار الأسهم")
        selected_stocks = st.multiselect("اختر الأسهم لتحليلها:", COMPANY_SYMBOLS, default=["AAPL", "MSFT", "TSLA"])
        if selected_stocks:
            fig = make_subplots(rows=2, cols=1, subplot_titles=('أسعار الأسهم', 'حجم التداول'), vertical_spacing=0.1)
            colors = px.colors.qualitative.Set1
            for i, symbol in enumerate(selected_stocks):
                color = colors[i % len(colors)]
                if symbol in yfinance_data and 'history' in yfinance_data[symbol]:
                    df = yfinance_data[symbol]['history']
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f"{symbol} السعر", line=dict(color=color, width=2)), row=1, col=1)
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name=f"{symbol} الحجم", marker_color=color, opacity=0.6), row=2, col=1)
                elif symbol in alpha_data:
                    df = alpha_data[symbol]
                    fig.add_trace(go.Scatter(x=df.index, y=df['4. close'], name=f"{symbol} السعر", line=dict(color=color, width=2)), row=1, col=1)
                    if '5. volume' in df.columns:
                        fig.add_trace(go.Bar(x=df.index, y=df['5. volume'], name=f"{symbol} الحجم", marker_color=color, opacity=0.6), row=2, col=1)
            fig.update_layout(height=600, title_text="لوحة تحليل الأسهم")
            fig.update_xaxes(title_text="التاريخ")
            fig.update_yaxes(title_text="السعر ($)", row=1, col=1)
            fig.update_yaxes(title_text="الحجم", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🏢 مقارنة القيمة السوقية للشركات")
        if fmp_data:
            market_caps = []
            companies = []
            sectors = []
            for symbol, data in fmp_data.items():
                if isinstance(data, dict) and data.get('mktCap'):
                    market_caps.append(data['mktCap'] / 1e9)
                    companies.append(symbol)
                    sectors.append(data.get('sector', 'Unknown'))
            if market_caps:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=companies, y=market_caps, text=[f"${cap:.1f} مليار" for cap in market_caps], textposition='auto', marker_color=px.colors.qualitative.Set3))
                fig.update_layout(title="مقارنة القيمة السوقية", xaxis_title="الشركة", yaxis_title="القيمة السوقية (مليار دولار)", height=500)
                st.plotly_chart(fig, use_container_width=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("أكبر شركة", companies[market_caps.index(max(market_caps))], f"${max(market_caps):.1f} مليار")
                with col2:
                    st.metric("أصغر شركة", companies[market_caps.index(min(market_caps))], f"${min(market_caps):.1f} مليار")
                with col3:
                    st.metric("متوسط القيمة السوقية", f"${np.mean(market_caps):.1f} مليار")

    with tab3:
        st.subheader("🌍 المؤشرات الاقتصادية (بيانات البنك الدولي)")
        if world_bank_data:
            indicator = st.selectbox("اختر المؤشر الاقتصادي:", ["الناتج المحلي", "التضخم", "البطالة"], key="wb_indicator")
            indicator_map = {"الناتج المحلي": "gdp", "التضخم": "inflation", "البطالة": "unemployment"}
            if indicator_map[indicator] in world_bank_data:
                data = world_bank_data[indicator_map[indicator]]
                if hasattr(data, 'reset_index'):
                    df = data.reset_index()
                    fig = go.Figure()
                    for country in df.columns[1:]:
                        if country in df.columns:
                            fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df[country], mode='lines+markers', name=country, line=dict(width=2)))
                    fig.update_layout(title=f"اتجاهات {indicator} حسب الدولة", xaxis_title="السنة", yaxis_title=f"قيمة {indicator}", height=500)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("بيانات البنك الدولي غير متاحة. تحقق من اتصالك بالإنترنت.")

    with tab4:
        st.subheader("💱 أسعار الصرف (بالدولار الأمريكي)")
        if exchange_rate_data and 'rates' in exchange_rate_data:
            rates = exchange_rate_data['rates']
            if rates:
                currencies = list(rates.keys())
                values = list(rates.values())
                fig = go.Figure(go.Bar(x=values, y=currencies, orientation='h', marker_color=px.colors.sequential.Viridis, text=[f"{rate:.3f}" for rate in values], textposition='auto'))
                fig.update_layout(title="أسعار الصرف الحالية (1 دولار = X عملة)", xaxis_title="سعر الصرف", yaxis_title="العملة", height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("لا تتوفر بيانات أسعار الصرف.")

    with tab5:
        st.subheader("🥇 سعر الذهب والسلع")
        if gold_price_data and 'price' in gold_price_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("سعر الذهب", f"${gold_price_data['price']:.2f}", f"{gold_price_data.get('currency', 'USD')} للأونصة")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=gold_price_data['price'], domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "سعر الذهب (دولار/أونصة)"}, gauge={'axis': {'range': [None, 2500]}, 'bar': {'color': "gold"},
                    'steps': [{'range': [0, 1500], 'color': "lightgray"}, {'range': [1500, 2000], 'color': "yellow"}, {'range': [2000, 2500], 'color': "orange"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 2200}}))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("بيانات سعر الذهب غير متاحة.")

    with st.sidebar:
        st.header("💬 سجل المحادثات")
        if st.session_state.conversation_history:
            for i, entry in enumerate(st.session_state.conversation_history):
                with st.expander(f"س{i+1}: {entry['question'][:50]}..."):
                    st.write(f"**السؤال**: {entry['question']}")
                    st.write(f"**الإجابة**: {entry['answer'][:200]}...")
                    if entry['sources']:
                        st.write(f"**المصادر**: {', '.join([doc.metadata.get('source', 'غير معروف') for doc in entry['sources']])}")
        if st.button("🔄 محادثة جديدة"):
            st.session_state.conversation_history = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()
        st.header("📊 مصادر البيانات")
        st.write("✅ Yahoo Finance")
        st.write("✅ واجهة برمجة البنك الدولي")
        st.write("✅ Alpha Vantage" if ALPHA_API_KEY else "⚠️ Alpha Vantage (مفتاح API مطلوب)")
        st.write("✅ Financial Modeling Prep" if FMP_API_KEY else "⚠️ FMP (مفتاح API مطلوب)")
        st.write("✅ أسعار الصرف" if OER_API_KEY else "⚠️ أسعار الصرف (بيانات وهمية)")
        st.write("✅ واجهة الذهب" if GOLDAP_API_KEY else "⚠️ واجهة الذهب (بيانات وهمية)")
        st.header("💡 نصيحة مالية")
        st.write(random.choice(tips))

    st.markdown("---")
    st.markdown("**مساعد الأسئلة المالية المحسن** - مدعوم بالذكاء الاصطناعي، Yahoo Finance، وبيانات البنك الدولي")

# تطبيق Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/risk-assessment')
def risk_assessment():
    return jsonify({"risk": "متوسط"})

@app.route('/api/recommendations')
def recommendations():
    return jsonify({"recommendation": random.choice(tips)})

@app.route('/api/portfolio')
def portfolio():
    return jsonify({"portfolio_analysis": "تم التحليل بنجاح"})

@app.route('/api/stock/<symbol>')
def stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1mo")
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        return jsonify(hist.tail(10).to_dict())
    except Exception as e:
        return jsonify({"error": str(e)})

# التشغيل الرئيسي
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="تشغيل تطبيق Streamlit أو Flask")
    parser.add_argument('--mode', choices=['streamlit', 'flask'], default='streamlit', help="اختر التطبيق للتشغيل")
    args = parser.parse_args()

    if args.mode == 'streamlit':
        st.write("جارٍ تشغيل تطبيق Streamlit...")
        run_streamlit()
    else:
        app.run(debug=True)
