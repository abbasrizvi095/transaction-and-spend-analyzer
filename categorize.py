from transformers import pipeline

# Zero-shot classifier (no fine-tuning needed for MVP)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

CATEGORIES = ["Food", "Utilities", "Travel", "Shopping", "Healthcare", "Others"]

def categorize_transaction(description):
    result = classifier(description, CATEGORIES)
    return result['labels'][0]  # Top predicted category
