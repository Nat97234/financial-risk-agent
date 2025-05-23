import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime

# ---------- واجهة المستخدم ----------
st.set_page_config(page_title="الوكيل الذكي للنصح المالي", layout="wide")
st.title("📊 وكيل تقييم المخاطر والنصح المالي الذكي")
st.markdown("""
هذا النظام يستخدم الذكاء الاصطناعي لتحليل الأسهم وتقديم نصائح مالية أولية.
أدخل رمز السهم (مثل `AAPL`, `TSLA`, `MSFT`) للحصول على معلومات تفصيلية.
""")

# ---------- إدخال المستخدم ----------
symbol = st.text_input("أدخل رمز السهم:", value="AAPL")

# ---------- جلب البيانات من yfinance ----------
def fetch_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")
        return info, hist
    except:
        return None, None

if symbol:
    info, hist = fetch_stock_info(symbol)
    if info:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"معلومات عن السهم: {symbol.upper()}")
            st.write(f"**السعر الحالي:** {info.get('currentPrice', 'N/A')} USD")
            st.write(f"**القيمة السوقية:** {info.get('marketCap', 'N/A'):,} USD")
            st.write(f"**بيتا:** {info.get('beta', 'N/A')}")
            st.write(f"**الأرباح:** {info.get('dividendRate', 'N/A')} USD")
            st.write(f"**متوسط الحجم:** {info.get('averageVolume', 'N/A'):,}")
            st.write(f"**تغير السهم اليوم:** {info.get('regularMarketChange', 0):.2f} USD")

        with col2:
            st.subheader("📈 الرسم البياني للسعر")
            st.line_chart(hist['Close'])

        # ---------- التحليل ----------
        st.markdown("---")
        st.subheader("📋 تحليل ذكي للسهم:")

        current_price = info.get("currentPrice")
        beta = info.get("beta")
        market_cap = info.get("marketCap")
        volume = info.get("averageVolume")
        change = info.get("regularMarketChange")
        dividend = info.get("dividendRate")

        analysis = f"""
كما هو الحال مع أي استثمار، هناك دائمًا درجة من المخاطر المرتبطة بالاستثمار في الأسهم، بما في ذلك سهم {symbol.upper()}.

- السعر الحالي للسهم هو **{current_price} دولارًا**.
- متوسط حجم التداول هو **{volume:,}**، مما يدل على نشاط تداول جيد.
- معدل بيتا هو **{beta}**، مما يعني أن السهم {"أكثر" if beta and beta > 1 else "أقل"} حساسية للتغيرات في السوق.
- القيمة السوقية للشركة هي **{market_cap:,} دولارًا**، مما يجعلها {"كبيرة" if market_cap and market_cap > 1e11 else "متوسطة"} الحجم.
- الأرباح المدفوعة هي **{dividend if dividend else "لا توجد أرباح مدفوعة"}**.
- التغير اليومي في السهم هو **{change:.2f} دولارًا**.

⚠️ هذا التحليل مبدئي ولا يُعتبر نصيحة استثمارية. يُرجى الرجوع لمصادر إضافية قبل اتخاذ قرار مالي.
"""
        st.markdown(analysis)

        # ---------- مثال على سؤال ذكي ----------
        st.markdown("---")
        st.subheader("🧠 مثال على سؤال ممكن تجربته:")
        st.markdown("""
        🔍 ما هي أبرز عوامل المخاطر في سهم Apple وهل يعتبر استثمارًا آمنًا في الوضع الحالي؟
        
        يمكنك طرح سؤالك بصيغة مشابهة في النسخة المتقدمة من الوكيل الذكي باستخدام نماذج لغوية أكبر.
        """)
    else:
        st.error("⚠️ لم يتم العثور على بيانات لهذا الرمز. يرجى التأكد من صحته.")