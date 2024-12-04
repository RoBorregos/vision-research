import subprocess
import json

# Define the prompt and model
prompt = input("Enter your prompt: ")
model = "llama3.2:1b"


# Run the Ollama CLI
process = subprocess.run(
    ["ollama", "run", model],
    input=prompt,
    text=True,
    capture_output=True
)

# Print the output
if process.returncode == 0:
    print("Model Response:", process.stdout.strip())
else:
    print("Error:", process.stderr.strip())
