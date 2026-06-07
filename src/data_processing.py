import os
import json
import random
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Base directory for relative paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure folders exist
os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "src"), exist_ok=True)

def generate_payment_clauses(n=600):
    payment_templates = [
        "Payment shall be made within {days} days of invoice date.",
        "The Buyer shall pay the Seller a sum of {amount} {currency} upon completion of services.",
        "Invoices are payable within {days} days from the date of receipt.",
        "All payments hereunder shall be made in {currency_name} via {payment_method}.",
        "A late payment fee of {percent}% per month will be charged on overdue balances.",
        "The client agrees to pay an initial deposit of {percent}% of the total contract value.",
        "Payments will be processed on a {frequency} basis following submission of reports.",
        "The hourly rate for the consultant's services is set at {amount} {currency} per hour.",
        "Billing will occur monthly, and payments must be remitted within {days} business days.",
        "Failure to make timely payments will result in a suspension of services.",
        "All amounts paid under this Agreement are non-refundable except as otherwise provided.",
        "The customer shall reimburse all reasonable travel and lodging expenses within {days} days.",
        "All fees are exclusive of taxes, which shall be paid by the customer.",
        "Payment of the final installment of {amount} {currency} is subject to final acceptance.",
        "The compensation package includes a base salary and a performance bonus of {amount} {currency}."
    ]
    
    days_options = [15, 30, 45, 60, 90]
    amounts = ["$1,000", "$5,000", "$10,000", "$25,000", "$50,000", "€2,500", "£5,000"]
    currencies = ["USD", "EUR", "GBP", "CAD"]
    currency_names = ["US Dollars", "Euros", "Great British Pounds"]
    methods = ["wire transfer", "credit card", "electronic funds transfer", "direct deposit"]
    percents = [1.5, 2, 5, 10, 20, 50]
    frequencies = ["monthly", "quarterly", "bi-weekly", "weekly"]
    
    clauses = []
    for _ in range(n):
        temp = random.choice(payment_templates)
        filled = temp.format(
            days=random.choice(days_options),
            amount=random.choice(amounts),
            currency=random.choice(currencies),
            currency_name=random.choice(currency_names),
            payment_method=random.choice(methods),
            percent=random.choice(percents),
            frequency=random.choice(frequencies)
        )
        clauses.append(filled)
    return clauses

def generate_liability_clauses(n=600):
    liability_templates = [
        "In no event shall either party be liable for any indirect, incidental, special, or consequential damages.",
        "The maximum aggregate liability of the Company under this Agreement shall not exceed the total fees paid.",
        "Neither party shall be liable to the other for any lost profits, revenue, or business opportunities.",
        "To the maximum extent permitted by law, liability for negligence is limited to {amount} {currency}.",
        "Each party shall indemnify, defend, and hold harmless the other party from any third-party claims.",
        "The limitation of liability set forth herein shall apply regardless of the form of action, whether in contract or tort.",
        "Disclosing Party's liability for breach of this agreement shall be capped at {amount} {currency}.",
        "Neither party excludes or limits its liability for death or personal injury caused by negligence.",
        "The Customer agrees that the Service Provider's total liability under this agreement is strictly limited.",
        "The indemnifying party shall pay all costs, damages, and attorney's fees finally awarded against the indemnified party.",
        "Except for breaches of confidentiality, neither party's liability shall exceed {amount} {currency}.",
        "The remedies provided in this Agreement are exclusive and in lieu of all other remedies.",
        "No action arising out of this Agreement may be brought by either party more than {years} years after the cause of action accrued.",
        "The limitations of liability in this section represent the agreed allocation of risk between the parties."
    ]
    
    amounts = ["$5,000", "$10,000", "$50,000", "$100,000", "$500,000", "€10,000", "£20,000"]
    currencies = ["USD", "EUR", "GBP"]
    years = [1, 2, 3, 5]
    
    clauses = []
    for _ in range(n):
        temp = random.choice(liability_templates)
        filled = temp.format(
            amount=random.choice(amounts),
            currency=random.choice(currencies),
            years=random.choice(years)
        )
        clauses.append(filled)
    return clauses

def generate_non_compete_clauses(n=600):
    non_compete_templates = [
        "The Employee shall not engage in any business that competes directly with the Employer for {months} months after termination.",
        "During the term of this Agreement and for {months} months thereafter, the Partner shall not solicit the Company's clients.",
        "The receiving party agrees not to compete with the disclosing party in the geographic region of {region}.",
        "The Consultant shall not solicit, recruit, or hire any employees of the Client for a period of {years} years.",
        "For a period of {months} months following termination, the Executive shall not work for any competitor in {region}.",
        "The Covenantor agrees not to establish, open, or operate a competing business within a {distance} mile radius.",
        "The Partner shall not solicit, induce, or attempt to influence any supplier or vendor to cease doing business with the Company.",
        "During the Restricted Period, the Employee will not render services to any Competing Enterprise.",
        "The restrictions in this Non-Compete section are acknowledged by both parties to be fair and reasonable.",
        "The restricted geographic area is defined as {region} and any surrounding territories.",
        "If the Employee violates this covenant, the Employer shall be entitled to seek injunctive relief and damages.",
        "The Seller covenants that it will not engage in a similar business to the one sold for a period of {years} years."
    ]
    
    months = [6, 12, 18, 24, 36]
    years = [1, 2, 3, 5]
    regions = ["the United States", "North America", "Europe", "the State of California", "the United Kingdom", "a 50-mile radius"]
    distance = [10, 25, 50, 100]
    
    clauses = []
    for _ in range(n):
        temp = random.choice(non_compete_templates)
        filled = temp.format(
            months=random.choice(months),
            years=random.choice(years),
            region=random.choice(regions),
            distance=random.choice(distance)
        )
        clauses.append(filled)
    return clauses

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML tags if any
    text = re.sub(r'<[^>]*>', '', text)
    # Keep alphanumeric characters, punctuation, and currency symbols
    text = re.sub(r"[^a-zA-Z0-9\s.,;:!?$€£%'\-\(\)]", "", text)
    return text.strip()

def tokenize(text):
    # Simple word level tokenization separating punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens

def build_vocab(tokenized_texts, max_vocab_size=5000):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    # Reserve index 0 for <PAD> and 1 for <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    # Add most common words to vocab
    most_common = counter.most_common(max_vocab_size - 2)
    for word, _ in most_common:
        vocab[word] = len(vocab)
        
    return vocab

def pad_sequence(tokens, vocab, max_len=50):
    # Convert tokens to indices, using <UNK> for OOV words
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    
    # Pad or truncate
    if len(indices) < max_len:
        # Pad with 0 (<PAD>)
        indices = indices + [vocab["<PAD>"]] * (max_len - len(indices))
    else:
        # Truncate
        indices = indices[:max_len]
        
    return indices

def process_dataset():
    jsonl_path = os.path.join(base_dir, "data", "contract_nli_v1.jsonl")
    
    print("Loading Contract-NLI dataset from JSONL...")
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            
    print(f"Loaded {len(rows)} raw rows.")
    
    # We will map each unique premise to a clause type based on entailment/contradiction of hypotheses
    # Let's group hypotheses by class
    termination_hyps = {
        "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
        "Some obligations of Agreement may survive termination of Agreement."
    }
    
    non_compete_hyps = {
        "Receiving Party shall not solicit some of Disclosing Party's representatives."
    }
    
    # Accumulate labels per premise
    premise_annotations = {}
    for row in rows:
        premise = row["premise"]
        hyp = row["hypothesis"]
        label = row["label"]
        subset = row["subset"]
        
        if premise not in premise_annotations:
            premise_annotations[premise] = {"labels": [], "subset": subset}
        
        if label in ["entailment", "contradiction"]:
            premise_annotations[premise]["labels"].append(hyp)
            
    # Classify premises based on annotations
    confidentiality_clauses = []
    termination_clauses = []
    non_compete_clauses_extracted = []
    other_clauses = []
    
    for premise, info in premise_annotations.items():
        labels = info["labels"]
        subset = info["subset"]
        
        if not labels:
            other_clauses.append((premise, "Other/None", subset))
        else:
            # Check if any label is in termination or non-compete
            has_term = any(h in termination_hyps for h in labels)
            has_non_compete = any(h in non_compete_hyps for h in labels)
            
            if has_term:
                termination_clauses.append((premise, "Termination Clause", subset))
            elif has_non_compete:
                non_compete_clauses_extracted.append((premise, "Non-Compete Clause", subset))
            else:
                confidentiality_clauses.append((premise, "Confidentiality Clause", subset))
                
    print(f"Extracted counts:")
    print(f"  Confidentiality: {len(confidentiality_clauses)}")
    print(f"  Termination: {len(termination_clauses)}")
    print(f"  Non-Compete (Extracted): {len(non_compete_clauses_extracted)}")
    print(f"  Other/None: {len(other_clauses)}")
    
    # Downsample and augment to balance classes
    # Target size is 600 per class
    target_size = 600
    
    random.shuffle(confidentiality_clauses)
    random.shuffle(termination_clauses)
    random.shuffle(other_clauses)
    
    confidentiality_sample = confidentiality_clauses[:target_size]
    termination_sample = termination_clauses[:target_size]
    other_sample = other_clauses[:target_size]
    
    # For Payment, Liability and Non-compete, we generate synthetic data
    payment_clauses = [(txt, "Payment Clause", random.choice(["train", "dev", "test"])) for txt in generate_payment_clauses(target_size)]
    liability_clauses = [(txt, "Liability Clause", random.choice(["train", "dev", "test"])) for txt in generate_liability_clauses(target_size)]
    
    # For Non-Compete, combine the extracted ones and add synthetic to reach 600
    num_non_compete_synth = target_size - len(non_compete_clauses_extracted)
    non_compete_synth = [(txt, "Non-Compete Clause", random.choice(["train", "dev", "test"])) for txt in generate_non_compete_clauses(num_non_compete_synth)]
    non_compete_sample = non_compete_clauses_extracted + non_compete_synth
    
    # Combine all classes
    all_data = confidentiality_sample + termination_sample + other_sample + payment_clauses + liability_clauses + non_compete_sample
    df = pd.DataFrame(all_data, columns=["text", "label", "subset"])
    
    print("\nProcessed Dataset Value Counts:")
    print(df["label"].value_counts())
    print("\nSubset Counts:")
    print(df["subset"].value_counts())
    
    # Save processed CSV
    df.to_csv(os.path.join(base_dir, "data", "processed_clauses.csv"), index=False, encoding='utf-8')
    print("\nSaved processed clauses to data/processed_clauses.csv")
    
    # Original dataset EDA stats (Task 1)
    # Total contracts in original Contract-NLI
    total_contracts_original = 607
    # Average length of original contracts is 1,913 words (from Stanford paper)
    avg_contract_length_words = 1913
    
    # Statistics of text lengths in our clause dataset
    df["char_len"] = df["text"].apply(len)
    df["word_len"] = df["text"].apply(lambda t: len(t.split()))
    
    avg_len = df["word_len"].mean()
    max_len = df["word_len"].max()
    min_len = df["word_len"].min()
    
    print(f"\n--- EDA Statistics (Task 1) ---")
    print(f"Total Contracts: {total_contracts_original}")
    print(f"Clause Types: {list(df['label'].unique())}")
    print(f"Average Clause Length (words): {avg_len:.2f}")
    print(f"Longest Clause (words): {max_len} (Text: '{df.loc[df['word_len'].idxmax(), 'text'][:100]}...')")
    print(f"Shortest Clause (words): {min_len} (Text: '{df.loc[df['word_len'].idxmin(), 'text']}')")
    
    # Save a stats file for references
    stats_dict = {
        "total_contracts": total_contracts_original,
        "clause_types": list(df['label'].unique()),
        "average_clause_length_words": avg_len,
        "longest_clause_words": int(max_len),
        "shortest_clause_words": int(min_len)
    }
    with open(os.path.join(base_dir, "data", "eda_stats.json"), "w") as sf:
        json.dump(stats_dict, sf, indent=2)
        
    # --- Generating Visualizations ---
    # 1. Clause Distribution
    plt.figure(figsize=(10, 5))
    df["label"].value_counts().plot(kind='bar', color='#4F46E5', edgecolor='black', alpha=0.8)
    plt.title("Clause Type Distribution in Processed Dataset", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Clause Type", fontsize=12, labelpad=10)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "assets", "clause_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Word Frequency (Top 25 words excluding common words/cleaning)
    all_words = []
    stopwords = {"the", "of", "and", "to", "in", "shall", "a", "be", "or", "this", "any", "that", "by", "with", "for", "as", "such", "agreement", "party", "either", "under", "on", "from", "other", "is"}
    for text in df["text"]:
        words = clean_text(text).split()
        for w in words:
            if w not in stopwords and len(w) > 2:
                all_words.append(w)
                
    word_counts = Counter(all_words)
    top_words = pd.DataFrame(word_counts.most_common(25), columns=["word", "count"])
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_words["word"], top_words["count"], color='#10B981', edgecolor='black', alpha=0.8)
    plt.title("Top 25 Most Frequent Words (Excluding Stopwords)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Word", fontsize=12, labelpad=10)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "assets", "word_frequency.png"), dpi=150)
    plt.close()
    
    # 3. Contract Length Histogram (Sentence/Clause length in words)
    plt.figure(figsize=(10, 5))
    plt.hist(df["word_len"], bins=30, color='#F59E0B', edgecolor='black', alpha=0.8)
    plt.title("Distribution of Clause Lengths (Word Count)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Words", fontsize=12, labelpad=10)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "assets", "contract_length_histogram.png"), dpi=150)
    plt.close()
    
    print("Visualizations generated and saved to assets/ directory.")
    
    # --- Text Engineering (Task 2) ---
    print("\n--- Performing Text Engineering ---")
    df["clean_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize)
    
    # Build vocabulary on training set only to prevent leakage
    train_tokens = df[df["subset"] == "train"]["tokens"]
    vocab = build_vocab(train_tokens, max_vocab_size=5000)
    print(f"Vocabulary built. Size: {len(vocab)} words.")
    
    # Save vocabulary
    with open(os.path.join(base_dir, "data", "vocab.json"), "w") as vf:
        json.dump(vocab, vf, indent=2)
    print("Saved vocabulary to data/vocab.json")
    
    # Pad sequences
    max_len = 50
    df["padded_indices"] = df["tokens"].apply(lambda tok: pad_sequence(tok, vocab, max_len=max_len))
    
    # Save the final preprocessed dataframe to a JSON file for model training (keeps lists intact)
    df.to_json(os.path.join(base_dir, "data", "preprocessed_data.json"), orient='records', indent=2)
    print("Saved preprocessed training-ready dataset to data/preprocessed_data.json")
    
if __name__ == "__main__":
    process_dataset()
