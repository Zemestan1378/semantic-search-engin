from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("🤖 Better Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = f"User: {user_input}\nBot:"

    result = chatbot(
        prompt,
        max_length=100,
        truncation=True
    )

    generated_text = result[0]["generated_text"]

    answer = generated_text.split("Bot:")[-1]

    print("Bot:", answer)