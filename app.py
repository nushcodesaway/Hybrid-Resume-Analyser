import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import extract_text, preprocess, skill_gap
from ml_models import get_hybrid_similarity, detect_anomalies, cluster_data
from behavior import generate_behavior

st.title("Hybrid Resume Analyzer 🚀")

uploaded_files = st.file_uploader("Upload resumes", accept_multiple_files=True)
query = st.text_input("Enter job description")

if uploaded_files and query:
    docs = []
    behaviors = []

    # 🔹 Extract + preprocess
    for file in uploaded_files:
        text = extract_text(file)
        text = preprocess(text)
        docs.append(text)
        behaviors.append(generate_behavior())

    # 🔹 ML pipeline
    scores = get_hybrid_similarity(docs, query)
    anomaly_df = detect_anomalies(behaviors)
    clusters = cluster_data(scores)

    # 🔹 Find best resume
    best_index = scores.argmax()

    # 🔹 Create table
    df = pd.DataFrame({
        "Filename": [f.name for f in uploaded_files],
        "Score (%)": [round(s * 100, 2) for s in scores],
        "Cluster": clusters,
        "Status": [
            "⚠️ Suspicious" if anomaly_df['anomaly'][i] == -1 else "✅ Normal"
            for i in range(len(uploaded_files))
        ]
    })

    df = df.sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

    st.subheader("📊 Ranked Results")
    st.dataframe(df)
    
    
    st.subheader("📊 Score Comparison")

    fig, ax = plt.subplots()
    ax.bar(df["Filename"], df["Score (%)"])
    ax.set_ylabel("Score (%)")

    st.pyplot(fig)

    st.success("Analysis Complete ✅")

    # 🔹 Detailed view
    st.subheader("📄 Detailed Results")

    for i, file in enumerate(uploaded_files):
        st.markdown(f"### {file.name}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Score", f"{scores[i]*100:.2f}%")
        col2.metric("Cluster", clusters[i])

        if anomaly_df['anomaly'][i] == -1:
            col3.metric("Status", "⚠️ Suspicious")
        else:
            col3.metric("Status", "✅ Normal")
        with st.expander("📡 Behavioral Data (IoT Simulation)"):
            st.write(behaviors[i])

        if i == best_index:
            st.write("🌟 BEST MATCH")

        st.divider()

    # 🔥 Explainability (Top Resume)
    st.subheader("🧠 Why this resume ranked #1")

    words = query.split()
    matched_words = [w for w in words if w in top_text]

    st.write("Matched keywords:", ", ".join(matched_words[:5]))

    # 🔥 Skill Gap Analysis
    matched, missing = skill_gap(top_text, query)

    st.subheader("📉 Skill Gap Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("✅ Matched Skills")
        st.write(", ".join(matched[:5]) if matched else "None")

    with col2:
        st.write("❌ Missing Skills")
        st.write(", ".join(missing[:5]) if missing else "None")
