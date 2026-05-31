from transformers import pipeline

# مدل سبک برای چت
chatbot = pipeline(
    "text-generation",
    model="gpt2"
)

print("🤖 Chatbot آماده است! برای خروج بنویس: exit")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    result = chatbot(user_input, max_length=80, num_return_sequences=1)

    reply = result[0]["generated_text"]

    print("Bot:", reply)