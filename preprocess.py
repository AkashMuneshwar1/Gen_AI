import os
import pdfplumber
import nltk
import json
from datasets import Dataset
from transformers import AutoTokenizer

nltk.download("stopwords")
from nltk.corpus import stopwords

# Folder containing PDFs
PDF_FOLDER = "pdfs"

# Extract text from all PDFs
def extract_text_from_pdfs():
    documents = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(PDF_FOLDER, file)) as pdf:
                text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                documents.append(text)
    return documents

# Preprocess text (remove stopwords)
def preprocess_text(documents):
    stop_words = set(stopwords.words("english"))
    cleaned_text = [" ".join([word for word in doc.split() if word.lower() not in stop_words]) for doc in documents]
    return cleaned_text

# Save data for training
def save_to_json(cleaned_text):
    data = [{"text": doc} for doc in cleaned_text]
    with open("train_data.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    docs = extract_text_from_pdfs()
    cleaned_docs = preprocess_text(docs)
    save_to_json(cleaned_docs)
    print(f"✅ Preprocessing complete. {len(cleaned_docs)} documents saved for training.")
