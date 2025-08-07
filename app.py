import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import urllib.parse
import requests
from bs4 import BeautifulSoup
import json
import html
#st.set_page_config(layout="wide")
st.title("🧬🧠 Epilepsy exome sequencing dashboard")
st.markdown(
    """
    <div style='text-align: center;'>
        <a href='https://rivaslab.stanford.edu' target='_blank'>
    <img src='https://mrivas.su.domains/gbe/wp-content/uploads/2025/01/gbe.png' width='120'/>
            <p style='font-size: 0.9em;'>Built by Rivas Lab</p>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


import requests
from bs4 import BeautifulSoup

def scrape_bing_images(query, max_results=3):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    params = {
        "q": query,
        "form": "QBLH"
    }
    response = requests.get("https://www.bing.com/images/search", headers=headers, params=params)
    soup = BeautifulSoup(response.text, "html.parser")
    
    image_data = []
    for tag in soup.find_all("a", class_="iusc"):
        m_json = tag.get("m")
        if m_json:
            try:
                metadata = json.loads(html.unescape(m_json))
                image_url = metadata.get("murl")
                title = metadata.get("t", query)
                if image_url:
                    image_data.append({
                        "url": image_url,
                        "title": title
                    })
                if len(image_data) >= max_results:
                    break
            except Exception:
                continue
    return image_data

# Load OpenAI key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def format_scientific(df, columns, sig_digits=2):
    df_formatted = df.copy()
    for col in columns:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.{sig_digits}e}")
    return df_formatted

# Specify the columns to convert
sci_cols = ["p_constraint", "p_pathogenicity", "p_unified", "p_missense", "p_pLoF"]

# --- Data Upload and Filtering ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, sep=None, engine='python')
    df["-log10_p_unified"] = -np.log10(df["p_unified"])
    return df

# --- File Upload with Fallback to Default ---
uploaded_file = st.file_uploader("Upload your exome variant summary file (.csv or .tsv)", type=["csv", "tsv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    source_label = "uploaded file"
    uploaded = 1
else:
    default_path = "epilepsy_variants_unified_model_pvalues.tsv"
    df = pd.read_csv(default_path, sep="\t")
    source_label = "default dataset (epilepsy_variants_unified_model_pvalues.tsv)"
    uploaded = 1

st.success(f"Loaded data from {source_label}")

if uploaded:
    df = load_data(uploaded_file)
    groups = df["group"].unique()
    selected_group = st.selectbox("Select group to visualize", groups)
    filtered_df = df[df["group"] == selected_group]
    filtered_df = filtered_df[filtered_df["p_unified"] < 0.001]
    filtered_df = filtered_df[filtered_df["n_variants"] > 15]
    st.subheader("Manhattan-style Plot")
    sorted_df = filtered_df.sort_values("-log10_p_unified", ascending=False)
    sorted_df = format_scientific(sorted_df, sci_cols)
    st.dataframe(sorted_df.reset_index(drop=True))
    fig = px.scatter(
        sorted_df,
        x="gene_name",
        y="-log10_p_unified",
        hover_data=["gene_id", "n_variants"],
        color="group",
        title=f"Manhattan Plot for Group: {selected_group}"
    )
    fig.update_layout(xaxis_title="Gene Name", yaxis_title="-log10(p_unified)")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gene Table")
    st.dataframe(sorted_df.reset_index(drop=True))

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
                model="gpt-4o",
                messages=st.session_state.chat_history,
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.subheader("🖼️ Gene Figure Finder (Unofficial Bing Scrape)")
    gene_query = st.text_input("Enter gene name to fetch figure")
    if st.button("Fetch figure") and gene_query:
        query = f"{gene_query} gene function diagram"
        images = scrape_bing_images(query, max_results=3)
        if images:
            for img in images:
                st.image(img["url"], caption=img["title"], width=300)
                st.markdown(f"[🔗 View image in browser]({img['url']})", unsafe_allow_html=True)
        else:
            st.warning("No images found.")
else:
    st.info("Please upload a gene result file to begin.")
