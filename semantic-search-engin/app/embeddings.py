from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded!")

text = "Artificial Intelligence"

embedding = model.encode(text)

print("Vector length:", len(embedding))
print("First 10 values:")
print(embedding[:10])