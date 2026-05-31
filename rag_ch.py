from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader

# مدل سبک و سریع
model = SentenceTransformer("all-MiniLM-L6-v2")

# دیتابیس vector
client = chromadb.Client()
collection = client.create_collection("docs")

# 1. خواندن PDF
pdf_path = input("PDF path: ")
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

# 2. chunk کردن ساده
chunks = text.split("\n")

# 3. embedding و ذخیره
for i, chunk in enumerate(chunks):
    if len(chunk.strip()) == 0:
        continue

    emb = model.encode(chunk).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[emb],
        documents=[chunk]
    )

print("\n✅ PDF loaded into AI memory!")

# 4. chat loop
while True:
    query = input("\nYou: ")

    if query == "exit":
        break

    q_emb = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    print("\nBot:", context)