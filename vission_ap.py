from transformers import pipeline
from PIL import Image

# ساخت pipeline
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

print("=== AI Vision System ===")

# مسیر تصویر
image_path = input("Enter image path: ")

# باز کردن تصویر
image = Image.open(image_path)

# تحلیل تصویر
result = captioner(image)
