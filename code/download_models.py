from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

# Install sentence-transformers if not available
try:
    import sentence_transformers
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers"])

print("Downloading DistilGPT-2 (lightweight model)...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print("DistilGPT-2 downloaded!")

print("Downloading lightweight embedding model...")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model downloaded!")

print("All models downloaded successfully!")
