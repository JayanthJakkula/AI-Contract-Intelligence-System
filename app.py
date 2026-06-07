import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add workspace src to path if needed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models import AttentionModel, PositionalEncoding
from data_processing import clean_text, tokenize, pad_sequence

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="AI Contract Intelligence System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Title Banner */
    .title-banner {
        background: linear-gradient(135deg, #4F46E5 0%, #10B981 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.2);
    }
    .title-banner h1 {
        margin: 0;
        font-size: 2.8rem;
    }
    .title-banner p {
        margin: 10px 0 0 0;
        font-size: 1.15rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #F8FAFC;
    }
    
    /* Clause highlights */
    .clause-card {
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 6px solid #CBD5E1;
        background-color: #F8FAFC;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .clause-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    }
    .clause-confidentiality {
        border-left-color: #3B82F6;
        background-color: #EFF6FF;
    }
    .clause-termination {
        border-left-color: #EF4444;
        background-color: #FEF2F2;
    }
    .clause-payment {
        border-left-color: #10B981;
        background-color: #ECFDF5;
    }
    .clause-liability {
        border-left-color: #F59E0B;
        background-color: #FFFBEB;
    }
    .clause-noncompete {
        border-left-color: #8B5CF6;
        background-color: #F5F3FF;
    }
    .clause-other {
        border-left-color: #6B7280;
        background-color: #F9FAFB;
    }
    
    .clause-header {
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .header-confidentiality { color: #1D4ED8; }
    .header-termination { color: #B91C1C; }
    .header-payment { color: #047857; }
    .header-liability { color: #B45309; }
    .header-noncompete { color: #6D28D9; }
    .header-other { color: #4B5563; }
    
    /* Word attention spans */
    .attn-word {
        padding: 2px 4px;
        margin: 2px;
        border-radius: 4px;
        display: inline-block;
        font-weight: 500;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Load Model and Vocabulary ---
@st.cache_resource
def load_resources():
    # Construct relative paths based on file location to support multi-platform execution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(base_dir, "data", "vocab.json")
    model_path = os.path.join(base_dir, "data", "attention_model.pt")
    comparison_path = os.path.join(base_dir, "data", "model_comparison.json")
    
    with open(vocab_path, "r", encoding='utf-8') as f:
        vocab = json.load(f)
        
    model = AttentionModel(
        vocab_size=len(vocab),
        embedding_dim=64,
        num_heads=4,
        num_classes=5,
        max_len=50
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    with open(comparison_path, "r", encoding='utf-8') as f:
        metrics = json.load(f)
        
    return vocab, model, metrics

try:
    vocab, model, metrics = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}. Please make sure you have run the training script successfully.")
    st.stop()

# Label mapping
label_map = {
    0: "Confidentiality Clause",
    1: "Termination Clause",
    2: "Payment Clause",
    3: "Liability Clause",
    4: "Non-Compete Clause"
}

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/law.png", width=80)
    st.markdown("## AI Contract Intelligence")
    st.markdown("Explore critical clause classification powered by custom **Sinusoidal Positional Encoding** and **Self-Attention** models built from scratch.")
    
    st.divider()
    
    # Model comparison metrics
    st.markdown("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Model**")
        st.write(f"Acc: `{metrics['baseline']['accuracy']:.3f}`")
        st.write(f"F1: `{metrics['baseline']['f1']:.3f}`")
    with col2:
        st.markdown("**Attention Model**")
        st.write(f"Acc: `{metrics['attention']['accuracy']:.3f}`")
        st.write(f"F1: `{metrics['attention']['f1']:.3f}`")
        
    st.divider()
    
    # Key Clause Definitions
    st.markdown("### 🔑 Target Clause Types")
    st.markdown("""
    🔵 **Confidentiality:** Definition of secrets, reverse engineering bans, sharing limits.
    🔴 **Termination:** Survival of terms, returns/destruction of info post-agreement.
    🟢 **Payment:** Invoicing schedules, late fees, deposits, compensation values.
    🟡 **Liability:** Caps on damages, indemnities, negligence limits, remedy exclusions.
    🟣 **Non-Compete:** Restrictions on soliciting clients/employees, geographical bans.
    """)

# --- Title Banner ---
st.markdown("""
<div class="title-banner">
    <h1>AI Contract Intelligence System</h1>
    <p>NLP + Self-Attention + Sinusoidal Positional Encoding from Scratch</p>
</div>
""", unsafe_allow_html=True)

# --- Main Layout ---
tab_predict, tab_compare = st.tabs(["🔍 Contract Clause Predictor", "📈 Positional & Attention Analysis"])

with tab_predict:
    st.markdown("### Upload or Paste Legal Contract")
    
    # Preset sample clauses for easy testing
    samples = {
        "Custom / Paste Your Own": "",
        "Sample NDA Excerpt (Confidentiality & Termination)": 
            "The Receiving Party shall not use any Confidential Information for any purpose other than the business relationship.\n"
            "All Confidential Information must be kept strictly secret and shall not be disclosed to any third party.\n"
            "This Agreement shall terminate three years from the date hereof.\n"
            "Upon termination, the Receiving Party shall destroy or return all copies of the Disclosing Party's Confidential Information within thirty days.",
        "Sample Service Agreement Excerpt (Payment & Liability)":
            "Payment shall be made within 30 days of receiving the monthly invoice.\n"
            "The Buyer shall pay the Seller the sum of $15,000 USD upon completion of the deliverables.\n"
            "A late payment interest of 2% per month will be charged on all outstanding balances.\n"
            "In no event shall either party be liable to the other for any indirect, incidental, or consequential damages.\n"
            "The aggregate liability of the consultant under this agreement is limited to the fees paid.",
        "Sample Employment Non-Compete Agreement":
            "The Employee shall not engage in any competing business for a period of 12 months after termination.\n"
            "During the restricted period, the Executive will not solicit or hire any employees or clients of the Employer.\n"
            "The Covenantor agrees not to operate a competing franchise within a 50-mile radius of the headquarters."
    }
    
    selected_sample = st.selectbox("Select a Sample Template:", list(samples.keys()))
    
    # Text input area
    default_text = samples[selected_sample]
    uploaded_file = st.file_uploader("Or upload a text contract (.txt)", type=["txt"])
    
    if uploaded_file is not None:
        contract_text = uploaded_file.read().decode("utf-8")
    else:
        contract_text = st.text_area("Paste contract text here:", value=default_text, height=180)
        
    if st.button("Run Clause Intelligence Analysis", type="primary"):
        if not contract_text.strip():
            st.warning("Please enter or upload some contract text first.")
        else:
            # Process sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', contract_text) if s.strip()]
            
            predictions = []
            
            for s in sentences:
                clean_txt = clean_text(s)
                tokens = tokenize(clean_txt)
                padded = pad_sequence(tokens, vocab, max_len=50)
                
                # Predict
                x = torch.tensor([padded], dtype=torch.long)
                with torch.no_grad():
                    logits, attn_weights = model(x)
                    pred_class = torch.argmax(logits, dim=1).item()
                    
                predictions.append({
                    "sentence": s,
                    "label": label_map[pred_class],
                    "tokens": tokens,
                    "attn": attn_weights.squeeze(0).mean(dim=0)[:len(tokens), :len(tokens)].numpy()
                })
                
            st.session_state["predictions"] = predictions
            st.success(f"Successfully analyzed {len(sentences)} clauses!")

    # Display Predictions
    if "predictions" in st.session_state:
        preds = st.session_state["predictions"]
        
        st.markdown("### 📋 Identified Clauses")
        st.write("Click on any clause below to inspect its word-level attention map and positional representations in the next tab.")
        
        # We will list each clause in a colored card
        for idx, item in enumerate(preds):
            label = item["label"]
            sentence = item["sentence"]
            
            # Map styling class
            style_class = "clause-other"
            header_class = "header-other"
            if label == "Confidentiality Clause":
                style_class = "clause-confidentiality"
                header_class = "header-confidentiality"
            elif label == "Termination Clause":
                style_class = "clause-termination"
                header_class = "header-termination"
            elif label == "Payment Clause":
                style_class = "clause-payment"
                header_class = "header-payment"
            elif label == "Liability Clause":
                style_class = "clause-liability"
                header_class = "header-liability"
            elif label == "Non-Compete Clause":
                style_class = "clause-noncompete"
                header_class = "header-noncompete"
                
            st.markdown(f"""
            <div class="clause-card {style_class}">
                <div class="clause-header {header_class}">Clause {idx+1}: {label}</div>
                <div style="font-size:1.05rem; color:#1E293B;">"{sentence}"</div>
            </div>
            """, unsafe_allow_html=True)


with tab_compare:
    st.markdown("### 📊 Attention & Positional Encoding Analysis")
    
    if "predictions" not in st.session_state:
        st.info("Please run the Clause Intelligence Analysis on the first tab to view the attention and positional encoding visualizations here.")
    else:
        preds = st.session_state["predictions"]
        
        # Dropdown to select sentence to analyze
        clause_options = {f"Clause {i+1}: {item['sentence'][:60]}...": i for i, item in enumerate(preds)}
        selected_clause_str = st.selectbox("Select a Clause to Inspect:", list(clause_options.keys()))
        selected_idx = clause_options[selected_clause_str]
        
        selected_item = preds[selected_idx]
        tokens = selected_item["tokens"]
        attn_weights = selected_item["attn"]
        label = selected_item["label"]
        sentence = selected_item["sentence"]
        
        # Word-Level Attention Highlight (Task 8: Highlight Important Terms)
        st.markdown("#### 🌟 Word Attention Highlighting")
        st.write("Words with higher self-attention weights are shaded darker. These represent the key terms the model focused on to make the classification.")
        
        # We compute the average attention score received by each word (sum of columns)
        word_scores = attn_weights.sum(axis=0)
        # Normalize scores for visualization alpha [0.1, 0.85]
        if len(word_scores) > 0 and word_scores.max() != word_scores.min():
            norm_scores = 0.1 + 0.75 * (word_scores - word_scores.min()) / (word_scores.max() - word_scores.min())
        else:
            norm_scores = [0.45] * len(word_scores)
            
        highlight_html = ""
        # Map class highlights colors
        color_hex = "#3B82F6" # blue
        if label == "Termination Clause":
            color_hex = "#EF4444" # red
        elif label == "Payment Clause":
            color_hex = "#10B981" # green
        elif label == "Liability Clause":
            color_hex = "#F59E0B" # orange
        elif label == "Non-Compete Clause":
            color_hex = "#8B5CF6" # purple
            
        for token, alpha in zip(tokens, norm_scores):
            # Convert hex to rgba
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            highlight_html += f'<span class="attn-word" style="background-color: rgba({r}, {g}, {b}, {alpha:.3f}); color: #0F172A;">{token}</span> '
            
        st.markdown(f'<div style="padding:1.5rem; background-color:#F8FAFC; border-radius:12px; font-size:1.25rem; line-height:2rem; border:1px solid #E2E8F0; margin-bottom:2rem;">{highlight_html}</div>', unsafe_allow_html=True)
        
        # Two columns for Heatmaps
        col_attn, col_pe = st.columns(2)
        
        with col_attn:
            st.markdown("#### 🗺️ Self-Attention Map (Task 8)")
            st.write("Shows how words in the clause attend to each other. Notice high self-interactions on nouns and verbs.")
            
            # Plot attention heatmap
            fig_attn, ax_attn = plt.subplots(figsize=(8, 6.5))
            sns.heatmap(
                attn_weights, 
                xticklabels=tokens, 
                yticklabels=tokens, 
                cmap="viridis", 
                annot=True if len(tokens) <= 12 else False,
                fmt=".2f",
                cbar_kws={'label': 'Attention weight'},
                ax=ax_attn
            )
            ax_attn.set_title("Self-Attention Alignment Matrix", fontsize=12, fontweight='bold', pad=10)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_attn)
            plt.close(fig_attn)
            
        with col_pe:
            st.markdown("#### 📈 Sinusoidal Positional Encoding Heatmap")
            st.write("Displays the mathematical positional vectors added to the word embeddings for the first N positions of this sentence.")
            
            # Reconstruct positional encodings for this sentence length
            pos_encoder = PositionalEncoding(d_model=64, max_len=50, dropout=0.0)
            pe_matrix = pos_encoder.pe.squeeze(0).cpu().numpy()[:len(tokens), :]
            
            fig_pe, ax_pe = plt.subplots(figsize=(8, 6.5))
            sns.heatmap(
                pe_matrix, 
                cmap="RdBu", 
                center=0, 
                cbar_kws={'label': 'Encoding value'},
                ax=ax_pe
            )
            ax_pe.set_title("Positional Encoding Grid (N x 64)", fontsize=12, fontweight='bold', pad=10)
            ax_pe.set_xlabel("Embedding Dimension", fontsize=10)
            ax_pe.set_ylabel("Word Position / Token Index", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_pe)
            plt.close(fig_pe)
            
        # Analysis summary
        st.divider()
        st.markdown("#### 📖 How does the Attention Model classify this clause?")
        st.markdown(f"""
        1. **Embedding & Position Injection:** Each token in the sentence (e.g., `"{tokens[0]}"` at index 0, `"{tokens[1]}"` at index 1) is projected into a 64-dimensional embedding space. A sinusoidal wave vector corresponding to its exact index is added to represent its structural order.
        2. **Multi-Head Self-Attention:** The words compute mutual query-key similarity matrices. Key legal markers like **{', '.join([t for t in tokens if t in ['payment', 'termination', 'confidential', 'liability', 'solicit', 'compete', 'days', 'months', 'overdue', 'limited']]) or 'key nouns/verbs'}** receive stronger weights.
        3. **Mean Pooling & Classification:** The context-rich outputs are averaged across the sentence and fed to the dense feed-forward layer to output logits for **`{label}`**.
        """)
