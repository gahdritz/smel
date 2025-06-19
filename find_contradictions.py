#Written with assistance from ChatGPT 4o
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load the semantic similarity model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the matched pairs file (CSV with columns like fake_text, true_text, etc.)
df = pd.read_csv("matched_articles.csv")  # Replace with your actual file path

# Optional: reduce size for testing
# df = df.sample(n=10, random_state=42)

results = []

for idx, row in df.iterrows():
    fake_sentences = [s.strip() for s in str(row['fake_text']).split('.') if s.strip()]
    true_sentences = [s.strip() for s in str(row['true_text']).split('.') if s.strip()]

    fake_embeds = model.encode(fake_sentences, convert_to_tensor=True)
    true_embeds = model.encode(true_sentences, convert_to_tensor=True)

    cos_sim = util.cos_sim(fake_embeds, true_embeds)

    fake_contras = set()
    true_contras = set()

    for i, fake_sent in enumerate(fake_sentences):
        for j, true_sent in enumerate(true_sentences):
            sim_score = cos_sim[i][j].item()
            if sim_score < 0.3:
                fake_contras.add(fake_sent)
                true_contras.add(true_sent)

    results.append({
        'fake_title': row['fake_title'],
        'fake_text': row['fake_text'],
        'fake_date': row['fake_date'],
        'fake_contradictions': list(fake_contras),
        'true_title': row['true_title'],
        'true_text': row['true_text'],
        'true_date': row['true_date'],
        'true_contradictions': list(true_contras)
    })

# Save results to new file
output_df = pd.DataFrame(results)
output_df.to_csv("contradiction_analysis_output.csv", index=False)

print("Contradiction analysis completed and saved to 'contradiction_analysis_output.csv'")
