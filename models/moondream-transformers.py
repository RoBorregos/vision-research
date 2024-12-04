'''
Moondream model with transformers
For prompt-based image captioning
'''

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# image = Image.open('random.png')
img_path = "random.png"
image = Image.open(img_path)
enc_image = model.encode_image(image)
print("Image Encoded")

while True:
    print(model.answer_question(enc_image, input("Enter your question: "), tokenizer))
