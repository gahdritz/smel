# Prompt for matching pairs of news articles

In order to recreate our real/fake news pairs, we use ChatGPT 4o to match potential sources between our `Fake.csv` and `Real.csv` articles.

The following prompts can be used with the web version in the below order, but specific subscription settings may require further optimization:

## Initial context
(Upload files `Fake.csv` and `Real.csv` along with this context)

```
I have two CSV files: Fake.csv and True.csv, each containing news articles with columns title, text, date, and subject.
```

## Preprocessing

```
Please preprocess both files, specifically:

1) Parse the date column as a datetime object.

2) Remove any rows with missing title or text.
```

## Matching

```
Match articles in the following manner:

1) Sample 1000 random articles from Fake.csv.

2) For each sampled fake article, find all articles in True.csv published within ±5 days.

3) Compute TF-IDF vectors using max_features=1000 on the text fields.

4) Calculate cosine similarity for each date-matching pair.

5) Retain only article pairs where similarity ≥ 0.7.

If needed, Use Segmented Expansion: Process 100-article batches incrementally and combine results.
``` 

## Table creation

``` 
Output a table named "Matched Articles (Full Sample of 1000, Score > 0.7)" with the following columns:

fake_title, fake_text, fake_date, true_title, true_text, true_date, similarity_score
```