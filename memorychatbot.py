from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("🤖 Chatbot with Memory Ready!")
print("Type 'exit' to quit.\n")

memory = ""

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # ذخیره مکالمه
    memory += f"User: {user_input}\nBot:"

    # ساخت prompt کامل
    result = chatbot(
        memory,
        max_length=200,
        pad_token_id=50256
    )

    output = result[0]["generated_text"]

    # فقط آخرین جواب را جدا می‌کنیم
    reply = output.split("Bot:")[-1].strip()

    print("Bot:", reply)

    # اضافه کردن جواب به حافظه
    memory += reply + "\n"