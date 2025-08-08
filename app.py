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
st.set_page_config(layout="wide")
def build_gene_prompt_from_df(df, thresh=6.5, fallback_n=20):
    # Pick genes passing threshold
    genes = (
        df.loc[df["-log10_p_unified"] > thresh, "gene_name"]
        .dropna().astype(str).unique().tolist()
    )
    # Fallback if none pass threshold
    if not genes:
        genes = (
            df.sort_values("-log10_p_unified", ascending=False)["gene_name"]
            .dropna().astype(str).head(fallback_n).tolist()
        )

    gene_csv = ", ".join(genes)

    prompt = (
        f"What role do {gene_csv} play in epilepsy? "
        "Provide references to the papers (do not include PMIDs). "
        "Include figures where appropriate. "
        "Describe what happens to loss-of-function variants and to gain-of-function variants. "
        "Provide therapeutic hypotheses (small molecule? antibody? gene therapy? genetic medicine?). "
        "Provide a full therapeutic-development matrix in table form for these genes. "
        "Describe therapeutic programs for each gene."
    )
    return prompt

st.title("🧬🧠 Epilepsy exome sequencing dashboard")
st.markdown(
    """
    <div style='text-align: left;'>
        <a href='https://rivaslab.stanford.edu' target='_blank'>
    <p style='font-size: 0.9em;'><img src='https://mrivas.su.domains/gbe/wp-content/uploads/2025/01/gbe.png' width='100'/>Built by Rivas Lab
       </p> </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Apply dark theme override manually
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stTextInput > div > input,
        .stSelectbox > div > div > div > input {
            background-color: #262730;
            color: #fafafa;
        }
        .stButton > button {
            background-color: #00c4b4;
            color: white;
        }
    </style>
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
    uploaded_file = "epilepsy_variants_unified_model_pvalues.tsv"
    df = pd.read_csv(uploaded_file, sep="\t")
    source_label = "default dataset (epilepsy_variants_unified_model_pvalues.tsv)"
    uploaded = 1

st.success(f"Loaded data from {source_label}")
st.markdown(f"**Data source**: `{source_label}`")
if uploaded_file:
    df = load_data(uploaded_file)
    groups = df["group"].unique()
    selected_group = st.selectbox("Select group to visualize", groups)
    filtered_df = df[df["group"] == selected_group]
    genesrm = ['TTN','PCDHGA11','PCDHAC2']
    filtered_df = filtered_df[~filtered_df['gene_name'].isin(genesrm)]
    filtered_df = filtered_df[filtered_df["p_unified"] < 0.001]
    filtered_df = filtered_df[filtered_df["n_variants"] > 15]
    st.subheader("Epilepsy results for analysis group: " + selected_group)
    sorted_df = filtered_df.sort_values("-log10_p_unified", ascending=False)
    sorted_df = format_scientific(sorted_df, sci_cols)
    st.dataframe(sorted_df.reset_index(drop=True))
    fig = px.scatter(
        sorted_df[sorted_df["-log10_p_unified"] >= 5],
        x="gene_name",
        y="-log10_p_unified",
        hover_data=["gene_id", "n_variants"],
        color="group",
        title=f"Manhattan Plot for Group: {selected_group}"
    )
    fig.update_layout(xaxis_title="Gene Name", yaxis_title="-log10(p_unified)")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")  # Horizontal divider

    st.markdown(
        "🧠 **Note:** Only genes with more than 15 variants were included in the unified meta regression model. [Paper](https://www.medrxiv.org/content/10.1101/2024.06.27.24309590v2) describing findings.",
        unsafe_allow_html=True
    )
    st.caption("Source of single-variant summary statistics: [Epi25](https://epi25.broadinstitute.org)")
    # --- AI Gene Interrogation Chat ---
    st.subheader("🔍 Gene AI Chat Assistant")
    if "last_group" not in st.session_state:
        st.session_state.last_group = selected_group
    if "gene_prompt" not in st.session_state:
        st.session_state.gene_prompt = build_gene_prompt_from_df(sorted_df, thresh=6.5)
    if selected_group != st.session_state.last_group:
        st.session_state.gene_prompt = build_gene_prompt_from_df(sorted_df, thresh=6.5)
        st.session_state.last_group = selected_group
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "gene_prompt" not in st.session_state:
        st.session_state.gene_prompt = DEFAULT_GENE_PROMPT
    prompt = st.text_area(
        "Prompt",
        value=st.session_state.gene_prompt,
        height=220,
        help="Edit this prompt or click Reset to restore the default."
    )
    b1, b2 = st.columns([1,1])
    with b1:
        ask = st.button("Ask")
    with b2:
        reset = st.button("Reset to default")
    if reset:
        st.session_state.gene_prompt = build_gene_prompt_from_df(sorted_df, thresh=6.5)
        st.rerun()
    if ask and prompt.strip():
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            try:
                stream = client.chat.completions.create(
                    model="gpt-5-chat-latest",
                    messages=st.session_state.chat_history,
                    stream=True,
                )
                reply = st.write_stream(stream)
            except Exception as e:
                st.error(f"Chat failed: {e}")
                reply = None
            if reply:
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
        # show prior messages (optional)
    with st.expander("Show conversation history", expanded=False):
        for msg in st.session_state.chat_history:
            who = "🧑‍🔬 You" if msg["role"] == "user" else "🤖 Assistant"
            st.markdown(f"**{who}:** {msg['content']}")
else:
    st.info("Please upload a gene result file to begin.")
