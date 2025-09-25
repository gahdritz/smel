#Written with assistance from ChatGPT 4o
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Select device: use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the semantic similarity model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load the matched pairs file (CSV with columns like fake_text, true_text, etc.)
df = pd.read_csv("Matched_Articles__Full_Sample_of_1000__Score___0_7_.csv")  # Replace with your actual file path

# Optional: reduce size for testing
# df = df.sample(n=10, random_state=42)

results = []

for idx, row in df.iterrows():
    fake_sentences = [s.strip() for s in str(row['fake_text']).split('.') if s.strip()]
    true_sentences = [s.strip() for s in str(row['true_text']).split('.') if s.strip()]

    # print(fake_sentences)

    fake_embeds = model.encode(fake_sentences, convert_to_tensor=True, device=device)
    true_embeds = model.encode(true_sentences, convert_to_tensor=True, device=device)

    cos_sim = util.cos_sim(fake_embeds, true_embeds)

    fake_contras = []
    true_contras = []

    contradiction_pairs = []

    for i, fake_sent in enumerate(fake_sentences):
        for j, true_sent in enumerate(true_sentences):
            sim_score = cos_sim[i][j].item()
            if sim_score < 0.1:
                fake_contras.append(fake_sent)
                true_contras.append(true_sent)

    results.append({
        'fake_title': row['fake_title'],
        'fake_text': row['fake_text'],
        'fake_date': row['fake_date'],
        'fake_contradictions': fake_contras,
        'true_title': row['true_title'],
        'true_text': row['true_text'],
        'true_date': row['true_date'],
        'true_contradictions': true_contras
    })

    print(results[0]['fake_contradictions'][0], results[0]['true_contradictions'][0])

    break

# Save results to new file
# output_df = pd.DataFrame(results)
# output_df.to_csv("contradiction_analysis_output.csv", index=False)

# print("Contradiction analysis completed and saved to 'contradiction_analysis_output.csv'")
