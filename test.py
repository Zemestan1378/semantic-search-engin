from transformers import pipeline

generator = pipeline("text-generation")

print(generator("Hello", max_new_tokens=20))