# AI/ML Transaction Grouper

This project groups raw financial transaction descriptions into meaningful semantic categories such as `Ride-hailing`, `Food Delivery`, `Streaming`, `Cloud Infrastructure`, `Payment Processing`, `Telecoms`, and `Retail / Grocery`.

It uses sentence embeddings with `all-MiniLM-L6-v2` and `KMeans` clustering to group similar transactions, then returns structured JSON with a label, grouped items, confidence score, explanation, and summary.

## Approach

I used a classical NLP pipeline:

1. Generate embeddings for each transaction description.
2. Cluster the embeddings with `KMeans`.
3. Map each cluster to a readable business label.
4. Compute confidence from average similarity within the cluster.
5. Return the result as structured JSON.

This approach works well for noisy transaction text because similar transactions often appear in different formats, abbreviations, or casing.

## Project Structure

```text
ai_ml_track/
├── data/
│   └── sample_input.json
├── src/
│   ├── embeddings.py
│   ├── grouper.py
│   └── main.py
├── tests/
│   └── test_grouper.py
└── README.md
```

## Why This Path

The assignment is not only about producing the right groups, but also about making clear and defensible modeling decisions.

I chose this path because it gives me:

- A semantic grouping method that is stronger than plain keyword matching
- A lightweight system that does not rely on external API access
- Predictable and reproducible behavior with fixed clustering settings
- A solution that is easy to debug and reason about

Using embeddings is important here because similar transaction descriptions often appear in very different surface forms. For example, `NETFLIX.COM 4500` and `Netflix subscription 4500` should be grouped together even though they are not exact string matches.

## Modeling Design

The implementation is organized into three main parts:

### 1. Embedding Generation

In `src/embeddings.py`, transaction descriptions are converted into vector embeddings using:

- `SentenceTransformer("all-MiniLM-L6-v2")`

The embeddings are normalized so cosine similarity can be used consistently.

### 2. Clustering

In `src/grouper.py`, the transaction embeddings are clustered using `KMeans`.

The current configuration fixes the number of clusters to `7`, matching the expected semantic categories in the dataset.

This is a reasonable choice for the take-home because the target grouping structure is known in advance, but it is also a limitation if the real input distribution changes.

### 3. Labeling and Output Formatting

Once clusters are formed:

- each cluster ID is mapped to a readable label such as `Ride-hailing` or `Streaming`
- an explanation is attached using a label-to-explanation dictionary
- confidence is derived from the average item similarity within the cluster
- output is formatted into the required JSON structure

## Assumptions

I made the following assumptions:

- The input is a flat list of transaction description strings
- Each transaction belongs to one primary semantic group
- The expected number of groups is known ahead of time
- The sample dataset is representative enough to support the current label mapping
- CPU-only inference is sufficient
- Cluster-center similarity is an acceptable heuristic for confidence

These assumptions make the system simpler and easier to explain, but they also limit how far it will generalize without further work.

## Trade-offs

What this solution gets right:

- It captures semantic similarity better than exact string matching
- It handles format variation such as casing, abbreviations, and merchant formatting differences
- It stays lightweight and API-free
- It produces structured, readable output aligned with the assessment requirements

What it may struggle with:

- `KMeans` requires the number of clusters to be chosen beforehand
- Cluster IDs are arbitrary, so mapping them to business labels is somewhat brittle
- The system may not handle unseen categories well
- There is no explicit outlier or ambiguity detection yet
- Confidence is heuristic rather than statistically calibrated

What would likely break first:

- Inputs containing more or fewer than the expected number of semantic categories
- Opaque or highly abbreviated bank references with little semantic signal
- Datasets containing many novel merchant types outside the current grouping pattern
- Situations where one merchant appears across multiple genuinely different categories with little text context

## Confidence Design

Confidence is computed from the average similarity score of transactions assigned to the same cluster.

Current thresholds are:

- `high` for average score >= `0.80`
- `medium` for average score >= `0.65`
- `low` for average score < `0.65`

This is intentionally simple and interpretable.

I do not treat this as a probability estimate. It is better understood as a group cohesion score: higher confidence means the items in a cluster are more semantically consistent with each other.

## Evaluation

There is no provided ground-truth labeled dataset, so I would evaluate the system using a mix of intrinsic and practical metrics.

### Without Labels

I would assess:

1. Cluster coherence through manual inspection
2. Intra-cluster similarity: items in the same group should be close
3. Inter-cluster separation: different groups should be meaningfully distinct
4. Stability across repeated runs with fixed seeds
5. Whether the generated labels and explanations are understandable to a reviewer

### With Labels

If labeled data were available, I would also measure:

1. Cluster purity
2. Adjusted Rand Index
3. Normalized Mutual Information
4. Per-category precision and recall after mapping clusters to labels

### Practical Proxy Check

A useful practical test would be perturbation testing:

- change casing
- remove punctuation
- abbreviate merchant names
- vary amount formatting

Then verify that semantically equivalent transactions still group together.

## Cost Estimate

This implementation uses a local embedding model and local clustering, so there is no direct API cost.

For 1,000 transactions:

- embedding generation is the dominant cost
- clustering cost is relatively small
- overall cost is effectively local compute only

That makes the solution inexpensive to run and appropriate for environments where external API access is unavailable or undesirable.

## Edge Cases

The current system partially handles the following cases:

- Different string formats for the same merchant
- Single-item groups
- Repeated transaction descriptions
- Semantically related items with different raw text

The current gaps are:

- no strong outlier detection
- no dedicated handling for ambiguous transactions
- no dynamic label generation for unseen clusters

## How To Run

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```