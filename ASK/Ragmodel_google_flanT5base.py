# Cell 1: install required libraries
!pip install -q transformers accelerate sentence-transformers faiss-cpu datasets
# If you have a GPU and want faster attention kernels you can install bitsandbytes + accelerate config (optional)
# !pip install -q bitsandbytes
#Imports and basic config
# Cell 2
import os
from pathlib import Path
import glob
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Paths
DOCS_DIR = Path("/content/docs")   # put your .txt files here (or mount Drive)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Model names
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # good general embeddings
GEN_MODEL = "google/flan-t5-large"  # generator
#Load or upload your dataset
# Cell 3 - utility to read docs from /content/docs
def load_docs(docs_dir):
    docs = []
    files = sorted(glob.glob(str(docs_dir/"*.txt")))
    for f in files:
        text = open(f, "r", encoding="utf-8").read().strip()
        if text:
            docs.append({"id": Path(f).stem, "text": text})
    return docs

docs = load_docs(DOCS_DIR)
print(f"Loaded {len(docs)} documents.")
# If you need an upload helper:
from google.colab import files
# files.upload()  # uncomment to upload files interactively
#Chunk documents (optional but recommended)
# Cell 4 - simple splitter
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# Build corpus of passages
passages = []
for doc in docs:
    chunks = chunk_text(doc["text"], chunk_size=200, overlap=50)
    for idx, c in enumerate(chunks):
        passages.append({"id": f"{doc['id']}_{idx}", "text": c})
print(f"Created {len(passages)} passages.")
#Create embeddings and FAISS index
# Cell 5 - embed passages
embed_model = SentenceTransformer(EMBED_MODEL)

texts = [p["text"] for p in passages]
batch_size = 32
embs = embed_model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

# Build FAISS index (inner product / cosine)
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)   # inner product
# normalize for cosine similarity
faiss.normalize_L2(embs)
index.add(embs)
print("FAISS index built with", index.ntotal, "vectors")

#Load Flan-T5 generator

## Cell 6 - load generator (tokenizer + model)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).cuda()  # move to GPU if available

# generation helper
def generate_from_prompt(prompt, max_new_tokens=128, num_beams=4):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.decode(out[0], skip_special_tokens=True)

#Retriever + prompt construction

# Cell 7 - retrieval function
def retrieve(query, top_k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < len(passages):
            results.append(passages[idx])
    return results

# prompt builder: simple concatenation (you can use templates)
def build_prompt(query, docs, prefix="Use the following context to answer the question. If the answer is not in the context, say you don't know.\n\n"):
    context = "\n\n---\n\n".join([d["text"] for d in docs])
    prompt = f"{prefix}\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    return prompt

#chat qna loop

# Cell 8 - interactive loop
def answer(query, top_k=5):
    docs = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, docs)
    ans = generate_from_prompt(prompt)
    return {"answer": ans, "retrieved": docs}

# Example usage
q = "What is the main idea of document 1?"  # replace with your query
res = answer(q, top_k=4)
print("Answer:\n", res["answer"])
print("\nRetrieved passages IDs:")
for d in res["retrieved"]:
    print("-", d["id"])



# Cell 9 - repeatable chat
print("Enter your question (empty to stop):")
while True:
    query = input("Q: ")
    if not query.strip():
        break
    out = answer(query, top_k=4)
    print("\nA:", out["answer"])
    print("Retrieved:", [d["id"] for d in out["retrieved"]])
    print("-"*40)





