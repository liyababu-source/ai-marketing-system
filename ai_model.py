import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("marketing_data.csv")

# Feature Engineering
df["CTR"] = df["Clicks"] / df["Impressions"]

# Conversion (for clustering)
df["Conversion"] = df["Clicks"] / df["Impressions"]

# ML Model (Audience / Conversion Prediction)
X = df[["Impressions", "Clicks"]]
y = (df["CTR"] > 0.05).astype(int)

model = LogisticRegression()
model.fit(X, y)

# K-Means Clustering (CORRECT WAY)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["CTR", "Conversion"]])

def run_ai(product, platform, impressions, clicks):

    # CTR Calculation
    ctr = clicks / impressions if impressions != 0 else 0

    # Conversion Prediction
    conversion_prob = model.predict_proba([[impressions, clicks]])[0][1]

    # Correct K-Means Prediction
    cluster = kmeans.predict([[ctr, conversion_prob]])[0]

    # Map cluster to segment
    if cluster == 0:
        segment = "High Value"
    elif cluster == 1:
        segment = "Medium Value"
    else:
        segment = "Low Value"

    # Audience Prediction
    if ctr > 0.07:
        audience = "Premium Shoppers"
    elif ctr > 0.03:
        audience = "Working Women"
    else:
        audience = "Youth"

    # Optimization Logic
    if conversion_prob > 0.7:
        decision = "Increase budget"
    elif conversion_prob > 0.3:
        decision = "Modify ad"
    else:
        decision = "Change audience"

    return {
        "product": product,
        "platform": platform,
        "audience": audience,
        "segment": segment,
        "ctr": round(ctr, 2),
        "conversion": round(conversion_prob, 2),
        "decision": decision
    }