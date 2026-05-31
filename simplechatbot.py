from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("🤖 Simple Chatbot Ready (type exit to quit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = f"User: {user_input}\nBot:"

    result = chatbot(
        prompt,
        max_length=80,
        pad_token_id=50256
    )

    text = result[0]["generated_text"]

    reply = text.split("Bot:")[-1]

    print("Bot:", reply)