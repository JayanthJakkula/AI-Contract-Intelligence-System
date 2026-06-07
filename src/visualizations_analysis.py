import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import PositionalEncoding, AttentionModel
from data_processing import clean_text, tokenize, pad_sequence

# Set styles for plots
sns.set_theme(style="white")

def task5_positional_encoding_visualization():
    print("\n--- Task 5: Positional Encoding Heatmap ---")
    d_model = 64
    max_len = 50
    
    # Initialize PositionalEncoding module
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    
    # Extract the registered buffer 'pe' which has shape (1, max_len, d_model)
    pe_matrix = pos_encoder.pe.squeeze(0).cpu().numpy() # shape: (50, 64)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pe_matrix, cmap="RdBu", center=0, cbar_kws={'label': 'Encoding Value'})
    plt.title("Sinusoidal Positional Encoding Heatmap (Task 5)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Embedding Dimension", fontsize=12, labelpad=10)
    plt.ylabel("Position", fontsize=12)
    plt.tight_layout()
    
    output_path = "d:/Deep_Learning/AI Contract Intelligence System/assets/positional_encoding_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Positional encoding heatmap saved to {output_path}")
    
    # Show individual dimensions for a few positions as requested: "Position 1, Position 2, Position 3, ..."
    plt.figure(figsize=(10, 4))
    positions_to_plot = [0, 1, 2, 3, 4]
    for pos in positions_to_plot:
        plt.plot(pe_matrix[pos, :20], label=f"Position {pos+1}") # plot first 20 dimensions for readability
    plt.title("Positional Encoding Vectors for Positions 1-5 (First 20 dimensions)", fontsize=12, fontweight='bold')
    plt.xlabel("Dimension", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    line_plot_path = "d:/Deep_Learning/AI Contract Intelligence System/assets/positional_encoding_lines.png"
    plt.savefig(line_plot_path, dpi=150)
    plt.close()
    print(f"Line plot for positions 1-5 saved to {line_plot_path}")


def task6_clause_understanding_analysis():
    print("\n--- Task 6: Clause Understanding Analysis ---")
    contract_a = "Payment shall be made within 30 days."
    contract_b = "Within 30 days payment shall be made."
    
    tokens_a = tokenize(clean_text(contract_a))
    tokens_b = tokenize(clean_text(contract_b))
    
    print(f"Contract A Tokens: {tokens_a}")
    print(f"Contract B Tokens: {tokens_b}")
    
    # We will look at positional encoding for word "payment"
    # Index of "payment" in A is 0, in B is 4
    idx_payment_a = tokens_a.index("payment")
    idx_payment_b = tokens_b.index("payment")
    
    # Index of "30" in A is 5, in B is 1
    idx_30_a = tokens_a.index("30")
    idx_30_b = tokens_b.index("30")
    
    d_model = 64
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=50, dropout=0.0)
    pe_matrix = pos_encoder.pe.squeeze(0).cpu().numpy()
    
    pe_payment_a = pe_matrix[idx_payment_a]
    pe_payment_b = pe_matrix[idx_payment_b]
    
    pe_30_a = pe_matrix[idx_30_a]
    pe_30_b = pe_matrix[idx_30_b]
    
    # Compute cosine similarities
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
    sim_payment = cosine_similarity(pe_payment_a, pe_payment_b)
    sim_30 = cosine_similarity(pe_30_a, pe_30_b)
    
    # Compute distance between "payment" and "30" in A vs B
    dist_a = np.linalg.norm(pe_matrix[idx_payment_a] - pe_matrix[idx_30_a])
    dist_b = np.linalg.norm(pe_matrix[idx_payment_b] - pe_matrix[idx_30_b])
    
    analysis_text = f"""Task 6: Clause Understanding Analysis
===================================
Contract A: "{contract_a}"
Contract B: "{contract_b}"

Tokens:
  Contract A: {tokens_a} (Length: {len(tokens_a)})
  Contract B: {tokens_b} (Length: {len(tokens_b)})

Analysis of Words at Different Positions:
1. The word 'payment' is identical in both contracts but occurs at different positions:
   - Contract A: Position {idx_payment_a + 1} (index {idx_payment_a})
   - Contract B: Position {idx_payment_b + 1} (index {idx_payment_b})
   - Cosine Similarity between their positional encodings: {sim_payment:.4f}

2. The word '30' also occurs at different positions:
   - Contract A: Position {idx_30_a + 1} (index {idx_30_a})
   - Contract B: Position {idx_30_b + 1} (index {idx_30_b})
   - Cosine Similarity between their positional encodings: {sim_30:.4f}

3. Distance in Sinusoidal Positional Space:
   - Euclidean distance between 'payment' and '30' positional representations in Contract A: {dist_a:.4f}
   - Euclidean distance between 'payment' and '30' positional representations in Contract B: {dist_b:.4f}

Explanation:
- Same Words, Different Order: Without positional encoding, a Bag of Words representation or standard embedding sum/average (like our Baseline Model) would treat both contracts identically. They contain the exact same set of tokens: {{'payment', 'shall', 'be', 'made', 'within', '30', 'days', '.'}}.
- Different Positional Representations: The sinusoidal positional encoding adds unique, position-dependent vectors to the word embeddings before self-attention is computed.
- Positional similarity (cosine similarity between 'payment' at index 0 vs index 4 is {sim_payment:.4f}) shows that the representations are mathematically distinct.
- Because the positional encodings are unique, the Self-Attention layer computes different dot products (Query-Key alignments) for 'payment' and '30' depending on where they appear. This enables the model to understand the exact structure and context of the clause (e.g., emphasis or main verb relationships), rather than just treating it as a flat set of words.
"""
    print(analysis_text)
    
    with open("d:/Deep_Learning/AI Contract Intelligence System/data/clause_analysis.txt", "w", encoding='utf-8') as f:
        f.write(analysis_text)
    print("Analysis saved to data/clause_analysis.txt")


def task7_attention_analysis():
    print("\n--- Task 7: Attention Analysis ---")
    
    # Load vocabulary
    with open("d:/Deep_Learning/AI Contract Intelligence System/data/vocab.json", "r", encoding='utf-8') as f:
        vocab = json.load(f)
        
    vocab_size = len(vocab)
    embedding_dim = 64
    num_heads = 4
    num_classes = 5
    max_len = 50
    
    # Initialize and load model
    model = AttentionModel(vocab_size, embedding_dim, num_heads, num_classes, max_len=max_len)
    model.load_state_dict(torch.load("d:/Deep_Learning/AI Contract Intelligence System/data/attention_model.pt", map_location=torch.device('cpu')))
    model.eval()
    
    # Sample sentence to analyze
    sample_sentence = "payment shall be made within 30 days of receiving the invoice."
    clean_txt = clean_text(sample_sentence)
    tokens = tokenize(clean_txt)
    
    # Pad sequence
    padded = pad_sequence(tokens, vocab, max_len=max_len)
    x = torch.tensor([padded], dtype=torch.long)
    
    # Forward pass to get attention weights
    # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
    _, attn_weights = model(x)
    
    # Remove batch dimension
    # attn_weights shape: (num_heads, seq_len, seq_len)
    attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()
    
    # Limit to the actual sequence length (tokens length) for visualization
    seq_len = len(tokens)
    
    # Average attention weights across all 4 heads
    # Shape becomes (seq_len, seq_len)
    mean_attn = attn_weights.mean(axis=0)[:seq_len, :seq_len]
    
    # Plot attention map
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=True, fmt=".2f", cbar_kws={'label': 'Attention Weight'})
    plt.title("Self-Attention Map for Sample Clause (Task 7)", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = "d:/Deep_Learning/AI Contract Intelligence System/assets/attention_map_sample.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Attention map sample saved to {output_path}")
    
    # Calculate most important words (words receiving the highest attention weights from other words)
    # This is the sum of attention weights along the columns (incoming attention)
    word_importance = mean_attn.sum(axis=0)
    
    print("\nMost Important Words based on Attention weights:")
    for idx, (token, score) in enumerate(zip(tokens, word_importance)):
        print(f"  Word: {token:<15} | Attention Score: {score:.4f}")
        
    # Save the attention statistics
    attn_stats = {
        "sentence": sample_sentence,
        "tokens": tokens,
        "importance_scores": word_importance.tolist()
    }
    with open("d:/Deep_Learning/AI Contract Intelligence System/data/attention_stats.json", "w", encoding='utf-8') as f:
        json.dump(attn_stats, f, indent=2)
    print("Attention statistics saved to data/attention_stats.json")

if __name__ == "__main__":
    task5_positional_encoding_visualization()
    task6_clause_understanding_analysis()
    task7_attention_analysis()
