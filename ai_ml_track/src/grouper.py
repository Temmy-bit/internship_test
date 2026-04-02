# from collections import defaultdict
import json
from sklearn.cluster import KMeans
from embeddings import get_embeddings, class_embeddings
import numpy as np
# import os

# Create Label With Cluster Id
def create_label(cluster_id):
    cluster_name_map = {
        0: "Payment Processing",
        1: "Food Delivery",
        2: "Streaming",
        3: "Cloud Infrastructure",
        4: "Ride-hailing",
        5: "Telecoms",
        6: "Retail / Grocery",
    }
    return cluster_name_map.get(cluster_id, "Other")


texts = [
    "Uber trip 1200",
    "UBER EATS ORDER 3400",
    "uber ride lagos 1100",
    "Netflix subscription 4500",
    "NETFLIX.COM 4500",
    "Amazon Web Services invoice",
    "AWS charges July",
    "Bolt ride 900",
    "BOLT TECHNOLOGIES 1050",
    "Paystack transfer fee",
    "Flutterwave payout 15000",
    "MTN airtime recharge 500",
    "MTN data bundle 1200",
    "Airtel subscription 800",
    "Shoprite purchase 3200",
    "Shoprite Lagos 2800",
]

embeddings = get_embeddings(texts)


kmeans = KMeans(n_clusters=7, random_state=26, n_init=10,algorithm="elkan")

cluster_labels = kmeans.fit_predict(embeddings)

clusters = {}
for text, cluster_id in zip(texts, cluster_labels):
    if cluster_id not in clusters:
        # print(f"Creating new cluster {create_label(cluster_id)} for text: '{text}'")
        clusters[cluster_id] = []
    clusters[cluster_id].append(text)

# Explanations for each category based on observed patterns
explanations = {
    "Ride-hailing": "Grouped by semantic similarity to transportation and ride services.",
    "Food Delivery": "Uber Eats is distinct from ride transport and belongs to food delivery.",
    "Streaming": "These items refer to the same streaming service with different raw text formats.",
    "Cloud Infrastructure": "AWS is the common acronym for Amazon Web Services.",
    "Payment Processing": "These transactions relate to payment processors and payout or transfer operations.",
    "Telecoms": "These items refer to telecom operators and mobile service purchases.",
    "Retail / Grocery": "These transactions refer to retail purchases from the same chain or retail context.",
}

#  Score to confidence mapping based on observed cosine similarity ranges
def score_to_confidence(score):
    if score >= 0.80:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"

# Calling class_embeddings to compute centroids for each category based on the grouped examples
# class_embeddings = class_embeddings(grouped_examples)

# Function to predict the category of a new transaction text based on cosine similarity to class centroids
def predict_class(text):
    emb = get_embeddings([text])[0]  # get single embedding
    
    best_label = None
    best_score = -1

    for label, class_emb in class_embeddings.items():
        score = np.dot(emb, class_emb)  # cosine similarity
        if score > best_score:
            best_score = score
            best_label = label

    confidence = score_to_confidence(best_score)

    return best_label, best_score, confidence


def grouped_output(test_texts, n_clusters=7):
    embeddings = get_embeddings(test_texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_

    grouped_data = {}

    for text, emb, cluster_id in zip(test_texts, embeddings, cluster_labels):
        center = cluster_centers[cluster_id]
        score = float(np.dot(emb, center) / (np.linalg.norm(emb) * np.linalg.norm(center)))

        label = f"{create_label(cluster_id)}"

        if label not in grouped_data:
            grouped_data[label] = {
                "label": label,
                "items": [],
                "confidence_scores": [],
                "explanation": f"{explanations.get(label)}",
            }

        grouped_data[label]["items"].append(text)
        grouped_data[label]["confidence_scores"].append(score)
    # print(list(explanations.keys()))
    groups = []
    for label, group in grouped_data.items():
        avg_score = float(np.mean(group["confidence_scores"]))
        groups.append({
            "label": label,
            "items": group["items"],
            "confidence": score_to_confidence(avg_score),
            "explanation": group["explanation"],
        })

    return groups, len(test_texts)



# Function to refine the grouped output by filtering out empty groups and preparing the final output format
def refine_output(groups, total_input):
    final_output = {
    "groups": groups,
    "ungrouped": [],
    "summary": {
        "total_input": total_input,
        "total_groups": len(groups),
        "ungrouped_count": 0
    }
    }
    final_output = json.dumps(final_output, indent=2)
    return final_output


if __name__ == "__main__":
    test_texts = [
        "Uber trip 1200",
        "UBER EATS ORDER 3400",
        "uber ride lagos 1100",
        "Netflix subscription 4500",
        "NETFLIX.COM 4500",
        "Amazon Web Services invoice",
        "AWS charges July",
        "Bolt ride 900",
        "BOLT TECHNOLOGIES 1050",
        "Paystack transfer fee",
        "Flutterwave payout 15000",
        "MTN airtime recharge 500",
        "MTN data bundle 1200",
        "Airtel subscription 800",
        "Shoprite purchase 3200",
        "Shoprite Lagos 2800"
    ]

    groups, test_length = grouped_output(test_texts)
    final_output = refine_output(groups, test_length)
    print(final_output)
