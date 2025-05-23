from flask import Flask, request, jsonify, render_template_string, send_file
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
from gtts import gTTS
import matplotlib.pyplot as plt
import logging
import warnings

# إخفاء جميع رسائل التحذير والـ logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.CRITICAL)
os.environ['PYTHONWARNINGS'] = 'ignore'

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
OER_API_KEY = os.getenv("EXCHANGERATES_API_KEY")

# Load local CSV data بدون إظهار أي رسائل تحذير
csv_path = 'financial_risk_analysis_large.csv'
try:
    financial_data = pd.read_csv(csv_path)
except (FileNotFoundError, Exception):
    # إنشاء DataFrame فارغ بصمت تام بدون أي رسائل
    financial_data = pd.DataFrame()

# Fetch stock data from Alpha Vantage
def get_stock_data(symbols=["AAPL"]):
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
        except Exception:
            # تجاهل جميع الأخطاء بصمت
            continue
    return all_data

alpha_data = get_stock_data()

# Convert all data into LangChain documents
def convert_data_to_documents(combined_data):
    documents = []
    # التأكد من وجود بيانات CSV قبل المعالجة
    if len(combined_data["csv_data"]) > 0:
        for row in combined_data["csv_data"][:100]:
            content = "\n".join([f"{k}: {v}" for k, v in row.items() if v is not None])
            documents.append(Document(page_content=content, metadata={"source": "csv"}))
    
    documents.append(Document(page_content=str(combined_data["stock_data_alpha"]), metadata={"source": "alpha_vantage"}))
    return documents

combined_data = {
    "csv_data": financial_data.to_dict(orient="records"),
    "stock_data_alpha": alpha_data
}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(convert_data_to_documents(combined_data))

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(split_docs, embeddings)

prompt_template = """
Use the following context to answer the question. Answer only using the information provided in the context.
If the answer is not present, say there's not enough info.
Respond in Arabic or English as asked.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <title>الوكيل المالي الذكي</title>
    <style>
        body { font-family: 'Arial'; background: #f4f4f4; padding: 20px; }
        .container { max-width: 600px; background: white; padding: 20px; border-radius: 10px; }
        textarea { width: 100%; height: 100px; }
        .result { margin-top: 20px; padding: 10px; background: #e8f4ea; border-radius: 5px; }
        button { padding: 10px 15px; margin-top: 10px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h2>وكيل تقييم المخاطر والنصح المالي الذكي</h2>
        <form method="POST">
            <textarea name="question" placeholder="اكتب سؤالك هنا..."></textarea>
            <button type="submit">إرسال</button>
        </form>
        {% if result %}
        <div class="result">
            <strong>الإجابة:</strong>
            <p>{{ result }}</p>
            <audio controls src="/audio"></audio>
            <br>
            <a href="/download-audio">تحميل الصوت</a>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        try:
            response = rag_chain.invoke({"query": question})
            result = response["result"]

            tts = gTTS(text=result, lang='ar' if any(c in question for c in 'ابتثجحخدذرزسشصضطظعغفقكلمنهوية') else 'en')
            tts.save("output.mp3")

            return render_template_string(HTML_TEMPLATE, result=result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template_string(HTML_TEMPLATE)

@app.route("/audio")
def audio():
    return send_file("output.mp3", mimetype="audio/mpeg")

@app.route("/download-audio")
def download_audio():
    return send_file("output.mp3", as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=True)
