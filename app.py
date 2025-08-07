import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Load OpenAI key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(layout="wide")
st.title("🧬 Exome Sequencing Dashboard")

def search_wikimedia_images(query, limit=10):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "iiprop": "url",
        "iiurlwidth": 300,
        "iiurlheight": 300,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    results = []
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        info = page.get("imageinfo", [{}])[0]
        if "thumburl" in info:
            results.append({
                "title": page.get("title", query),
                "thumburl": info["thumburl"],
                "pageurl": info["descriptionurl"]
            })
    return results

# --- Data Upload and Filtering ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, sep=None, engine='python')
    df["-log10_p_unified"] = -np.log10(df["p_unified"])
    return df

uploaded_file = st.file_uploader("Upload gene result file", type=[".csv", ".tsv"])
if uploaded_file:
    df = load_data(uploaded_file)
    groups = df["group"].unique()
    selected_group = st.selectbox("Select group to visualize", groups)
    filtered_df = df[df["group"] == selected_group]
    filtered_df = filtered_df[filtered_df["p_unified"] < 0.001]
    st.subheader("Manhattan-style Plot")
    fig = px.scatter(
        filtered_df,
        x="gene_name",
        y="-log10_p_unified",
        hover_data=["gene_id", "n_variants"],
        color="group",
        title=f"Manhattan Plot for Group: {selected_group}"
    )
    fig.update_layout(xaxis_title="Gene Name", yaxis_title="-log10(p_unified)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gene Table")
    st.dataframe(filtered_df.reset_index(drop=True))

    # --- AI Gene Interrogation Chat ---
    st.subheader("🔍 Gene AI Chat Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about a gene and group (e.g., What is DPM1's role in DEE?)"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.chat_history,
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.subheader("🖼️ Gene Figure Finder (Wikimedia Commons)")
    gene_query = st.text_input("Enter gene name to fetch diagram")
    if st.button("Fetch figure") and gene_query:
        images = search_wikimedia_images(f"{gene_query} gene diagram")
        if images:
            for img in images:
                st.image(img["thumburl"], caption=img["title"], width=300)
                st.markdown(f"[🔗 Source Page]({img['pageurl']})", unsafe_allow_html=True)
        else:
            st.warning("No images found.")
else:
    st.info("Please upload a gene result file to begin.")
