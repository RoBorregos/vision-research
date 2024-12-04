'''
Moondream model with Ollama
For prompt-based image captioning
'''

import subprocess
import json

# Define the model, input image, and instruction
model = "moondream"
image_path = "random.png"
prompt = "What is the woman wearing, what is she doing, and how does the image feel?"

# Format the input payload as a JSON string
input_payload = json.dumps({
    "model": model,
    "image_path": image_path,
    "prompt": prompt
})

# Run the Ollama CLI for the Moondream model
process = subprocess.run(
    ["ollama", "run", model],
    input=input_payload,
    text=True,
    capture_output=True
)

# Print the output or error
if process.returncode == 0:
    print("Model Response:", process.stdout.strip())
else:
    print("Error:", process.stderr.strip())
