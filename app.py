
import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# Load OpenAI key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(layout="wide")
st.title("🧬 Exome Sequencing Dashboard")

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
                model="gpt-4",
                messages=st.session_state.chat_history,
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # --- Figure Fetcher ---
    st.subheader("🖼️ Gene Role Figure Finder")
    gene_query = st.text_input("Enter gene name to fetch biological figure")
    if st.button("Fetch figure") and gene_query:
        search_url = "https://www.google.com/search"
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {"q": f"{gene_query} gene function diagram", "tbm": "isch"}
        response = requests.get(search_url, headers=headers, params=params)
        soup = BeautifulSoup(response.text, "html.parser")
        img_tag = soup.find("img")
        if img_tag and img_tag.get("src"):
            st.image(img_tag["src"], caption=f"Figure for {gene_query}")
        else:
            st.error("No figure found.")
else:
    st.info("Please upload a gene result file to begin.")
