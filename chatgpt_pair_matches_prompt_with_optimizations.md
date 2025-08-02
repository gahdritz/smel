### **Enhanced Prompt with Performance Optimizations**
CSV Files were collected from: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
I have two CSV files: `Fake.csv` and `True.csv`, each containing news articles with the columns: `title`, `text`, `date`, and `subject`.

Please perform the following steps to identify similar article pairs, applying optimizations to improve processing efficiency and scalability:

---

### **Preprocessing**

1. **Parse the `date` column** in both files as `datetime` objects.
2. **Remove any rows** where either the `title` or `text` fields are missing or null.
3. **Ensure only rows with valid parsed dates** are retained.

---

### **Similarity Matching Workflow**

1. **Randomly sample 5,000 articles** from the cleaned `Fake.csv` in increments of 500 without repetition.

2. For each sampled fake article:

   * Identify all articles from `True.csv` whose publication date is within a **±5-day window**.

3. To compute textual similarity:

   * Use **TF-IDF vectorization** on the `text` fields with `max_features=1000`.
   * **Fit the TF-IDF vectorizer once** on the combined corpus of all true articles and the sampled fake articles to prevent repeated re-fitting.
   * **Transform all true article texts in advance** and cache their TF-IDF vectors for reuse.

4. For each date-matching fake-true article pair:

   * Transform the fake article's text using the **pre-fitted TF-IDF vectorizer**.
   * Calculate **cosine similarity** between the fake vector and each matched true article vector.

5. **Retain only those article pairs** where cosine similarity is **≥ 0.7**.

6. Consolidate all pairs into a single results table, totaling the analysis of **5,000 unique fake articles**

---

### **Performance Optimizations**

* Apply **Segmented Expansion**: Process fake articles in **small batches (e.g., 20 articles at a time)** to stay within memory and execution limits.
* Avoid repeated vectorizer training by **precomputing and caching TF-IDF vectors** for all `True.csv` articles.
* Only compute similarity where a valid date-based match exists to minimize unnecessary computation.

---


### **Table creation**

Output a table named "Matched Articles (Full Sample of 5000, Score > 0.7)" with the following columns:

`fake_title`, `fake_text`, `fake_date`, `true_title`, `true_text`, `true_date`, `similarity_score`

### **Deduplication**

Remove any duplicate fake-true article pairs, where the same combination of `fake_title`, `fake_date`, `true_title`, and `true_date` appears more than once in the results

