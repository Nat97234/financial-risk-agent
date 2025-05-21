import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
import plotly.express as px

# إعداد الصفحة
st.set_page_config(page_title="تحليل المخاطر المالية", layout="wide")
st.title("📊 تحليل المخاطر المالية")
st.markdown("مرحبًا! هذا التطبيق يتيح لك استكشاف البيانات المالية وطرح أسئلة اقتصادية باستخدام نظام RAG.")

# تحميل مفاتيح API
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")

# دالة لجلب بيانات الأسهم
def get_stock_data(symbols=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "JPM", "WMT", "PG"]):
    all_data = {}
    for symbol in symbols:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_API_KEY}&outputsize=compact"
        try:
            response = requests.get(url)
            data = response.json()
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float)
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                all_data[symbol] = df
            else:
                st.warning(f"لا توجد بيانات متاحة لـ {symbol}. تحقق من مفتاح API.")
        except Exception as e:
            st.error(f"خطأ في جلب بيانات {symbol}: {e}")
    return all_data

# دالة لجلب ملفات تعريف الشركات
def get_company_profile(symbols):
    all_data = {}
    for symbol in symbols:
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FMP_API_KEY}"
        try:
            response = requests.get(url)
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                all_data[symbol] = data[0]
        except Exception as e:
            st.error(f"خطأ في ملف الشركة {symbol}: {e}")
    return all_data

# دالة أسعار الصرف
def get_exchange_rate(base="USD", symbols="EUR"):
    url = f"https://openexchangerates.org/api/latest.json?app_id={OER_API_KEY}&base={base}&symbols={symbols}"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        st.error(f"خطأ في أسعار الصرف: {e}")
        return {}

# بيانات World Bank
def get_worldbank_inflation_data(country="USA", indicator="FP.CPI.TOTL.ZG", start_year=2000, end_year=2024):
    url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&date={start_year}:{end_year}"
    try:
        response = requests.get(url)
        data = response.json()
        if data and len(data) > 1 and isinstance(data[1], list):
            df = pd.DataFrame(data[1])
            df['date'] = pd.to_datetime(df['date'])
            df['country'] = df['country'].apply(lambda x: x['value'])
            df['indicator'] = df['indicator'].apply(lambda x: x['value'])
            return df[['date', 'value', 'country', 'indicator']].dropna()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"خطأ World Bank: {e}")
        return pd.DataFrame()

# بيانات Eurostat
def get_eurostat_inflation_data(dataset="prc_hicp_manr", geo="DE", start_period="2010-01"):
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?format=JSON&geo={geo}&unit=RT12&startPeriod={start_period}"
    try:
        response = requests.get(url)
        data = response.json()
        if 'value' in data and 'dimension' in data:
            dates = list(data['dimension']['time']['category']['index'].keys())
            values = [float(data['value'].get(str(i), None)) for i in range(len(dates)) if str(i) in data['value']]
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'value': values,
                'indicator': 'HICP Annual Rate',
                'region': geo
            }).dropna()
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"خطأ Eurostat: {e}")
        return pd.DataFrame()

# تحميل البيانات
@st.cache_data
def load_data():
    try:
        csv_path = 'financial_risk_analysis_large.csv'
        financial_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error("الملف غير موجود.")
        financial_data = pd.DataFrame()
    
    alpha_data = get_stock_data()
    fmp_data = get_company_profile(list(alpha_data.keys()))
    exchange_rate = get_exchange_rate()
    worldbank = get_worldbank_inflation_data()
    eurostat = get_eurostat_inflation_data()
    
    return {
        "csv_data": financial_data.to_dict(orient="records"),
        "stock_data_alpha": alpha_data,
        "company_profile_fmp": fmp_data,
        "exchange_rate": exchange_rate,
        "worldbank_inflation": worldbank.to_dict(orient="records"),
        "eurostat_inflation": eurostat.to_dict(orient="records")
    }
url = "https://huggingface.co/datasets/Naat97/financial-risk-data/tree/main"
df = pd.read_csv(url)

# تحويل البيانات إلى مستندات
def convert_data_to_documents(combined_data):
    documents = []
    for row in combined_data["csv_data"][:100]:
        content = "\n".join([f"{k}: {v}" for k, v in row.items() if pd.notnull(v)])
        documents.append(Document(page_content=content, metadata={"source": "csv"}))
    
    documents.append(Document(page_content=str(combined_data["stock_data_alpha"]), metadata={"source": "stocks"}))

    fmp_text = "\n".join([f"{s}: {', '.join([f'{k}: {v}' for k, v in p.items()])}" for s, p in combined_data["company_profile_fmp"].items()])
    documents.append(Document(page_content=fmp_text, metadata={"source": "companies"}))
    
    exchange_text = f"USD/EUR Exchange Rate: {combined_data['exchange_rate'].get('rates', {}).get('EUR', 'N/A')}"
    documents.append(Document(page_content=exchange_text, metadata={"source": "forex"}))

    for row in combined_data["worldbank_inflation"]:
        documents.append(Document(page_content=f"{row['country']} inflation: {row['value']}% in {row['date'].year}", metadata={"source": "worldbank"}))
    
    for row in combined_data["eurostat_inflation"]:
        documents.append(Document(page_content=f"{row['region']} inflation: {row['value']}% in {row['date'].strftime('%Y-%m')}", metadata={"source": "eurostat"}))

    return documents

# إعداد RAG
@st.cache_resource
def setup_rag(combined_data):
    documents = convert_data_to_documents(combined_data)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    prompt = PromptTemplate(
        template="""استخدم المعلومات التالية للإجابة على السؤال التالي:\n\nالسياق:\n{context}\n\nالسؤال:\n{question}\n\nالإجابة بالعربية:""",
        input_variables=["context", "question"]
    )

    llm = OpenAI(model_name="text-davinci-003", openai_api_key=OPENAI_API_KEY)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return rag_chain

# واجهة التطبيق
combined_data = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["البيانات المالية", "ملفات الشركات", "التضخم", "استفسارات RAG"])

with tab1:
    st.header("البيانات المالية")
    df = pd.DataFrame(combined_data["csv_data"])
    st.dataframe(df.head())

with tab2:
    st.header("ملفات الشركات")
    for symbol, profile in combined_data["company_profile_fmp"].items():
        with st.expander(symbol):
            st.write(profile)

with tab3:
    st.header("بيانات التضخم")
    df_wb = pd.DataFrame(combined_data["worldbank_inflation"])
    df_eu = pd.DataFrame(combined_data["eurostat_inflation"])
    if not df_wb.empty:
        st.line_chart(df_wb.set_index("date")["value"])
    if not df_eu.empty:
        st.line_chart(df_eu.set_index("date")["value"])

with tab4:
    st.header("اسأل النظام")
    query = st.text_input("أدخل سؤالك:")
    if query:
        with st.spinner("جارٍ المعالجة..."):
            rag_chain = setup_rag(combined_data)
            result = rag_chain.invoke({"query": query})
            st.success("الإجابة:")
            st.write(result["result"])
