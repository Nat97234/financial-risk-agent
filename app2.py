from flask import Flask, request, jsonify, render_template_string
import os
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Setup Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")

# Load local CSV data
csv_path = 'financial_risk_analysis_large.csv'
financial_data = pd.read_csv(csv_path)

# Fetch stock data from Alpha Vantage
def get_stock_data(symbols=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]):
    all_data = {}
    for symbol in symbols:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol= {symbol}&apikey={ALPHA_API_KEY}&outputsize=compact"
        try:
            response = requests.get(url)
            data = response.json()
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index').astype(float)
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                all_data[symbol] = df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return all_data

alpha_data = get_stock_data()

# Fetch company profiles from FMP
def get_company_profile(symbols=["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]):
    all_data = {}
    for symbol in symbols:
        url = f"https://financialmodelingprep.com/api/v3/profile/ {symbol}?apikey={FMP_API_KEY}"
        try:
            response = requests.get(url)
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                all_data[symbol] = data[0]
        except Exception as e:
            print(f"Error fetching profile data for {symbol}: {e}")
    return all_data

fmp_data = get_company_profile()

# Fetch exchange rate
def get_exchange_rate(base="USD", symbols="EUR"):
    url = f"https://openexchangerates.org/api/latest.json?app_id= {OER_API_KEY}&base={base}&symbols={symbols}"
    response = requests.get(url)
    try:
        return response.json()
    except:
        return {}

exchange_rate_data = get_exchange_rate()

# Convert all data into LangChain documents
def convert_data_to_documents(combined_data):
    documents = []
    for row in combined_data["csv_data"][:100]:
        content = "\n".join([f"{k}: {v}" for k, v in row.items() if v is not None])
        documents.append(Document(page_content=content, metadata={"source": "csv"}))
    documents.append(Document(page_content=str(combined_data["stock_data_alpha"]), metadata={"source": "alpha_vantage"}))
    documents.append(Document(page_content=str(combined_data["company_profile_fmp"]), metadata={"source": "fmp"}))
    documents.append(Document(page_content=str(combined_data["exchange_rate"]), metadata={"source": "exchange_rates"}))
    return documents

# Combine all data
combined_data = {
    "csv_data": financial_data.to_dict(orient="records"),
    "stock_data_alpha": alpha_data,
    "company_profile_fmp": fmp_data,
    "exchange_rate": exchange_rate_data
}

# Convert to documents
documents = convert_data_to_documents(combined_data)

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Define prompt template
prompt_template = """
Use the following context to answer the question. Answer **only** using the information provided in the context.
If the answer is not present in the context, clearly say that there is not enough information to answer.
Respond in the same language as the question — Arabic or English.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Build RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Serve HTML page and handle questions
HTML_TEMPLATE = open("index.html", "r", encoding="utf-8").read()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        try:
            response = rag_chain.invoke({"query": question})
            return jsonify({
                "result": response["result"],
                "source_documents": [doc.page_content for doc in response["source_documents"]]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=True)