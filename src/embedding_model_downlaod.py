# Run this script ONCE in an environment with internet access
from sentence_transformers import SentenceTransformer
import os

model_name_to_download = "all-MiniLM-L6-v2"
local_path = "data/embedding_models" # This matches LOCAL_EMBEDDING_MODEL_PATH in settings.py

os.makedirs(local_path, exist_ok=True)
# This downloads the model weights and saves them to the specified path
model = SentenceTransformer(model_name_to_download)
model.save(os.path.join(local_path, model_name_to_download))
print(f"Model '{model_name_to_download}' downloaded to '{os.path.join(local_path, model_name_to_download)}'")