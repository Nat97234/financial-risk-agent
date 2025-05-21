import gdown

url = "https://drive.google.com/uc?id=1crVv8mcnL_uT0fAZ_8aINgJXLK-6VL3r"
output = "financial_risk_analysis_large.csv"
gdown.download(url, output, quiet=False)
