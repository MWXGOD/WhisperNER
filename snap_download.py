from huggingface_hub import snapshot_download


snapshot_download(repo_id="openai/whisper-small", local_dir = "cache/whisper-small", ignore_patterns=["tf_model.h5", "flax_model.msgpack", "model.safetensors"], resume_download=True)
