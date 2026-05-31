from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("Chatbot Ready")

while True:
    text = input("You: ")

    if text == "exit":
        break

    result = chatbot(text, max_length=60)

    print("Bot:", result[0]["generated_text"])