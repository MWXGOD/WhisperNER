from huggingface_hub import snapshot_download


# snapshot_download(repo_id="openai/whisper-small", local_dir = "cache/whisper-small", ignore_patterns=["tf_model.h5", "flax_model.msgpack", "model.safetensors"], resume_download=True)

snapshot_download(repo_id="openai/whisper-large-v3", local_dir = "cache/whisper-large-v3", ignore_patterns=["tf_model.h5", "flax_model.msgpack", "model.safetensors", "model.fp32-00001-of-00002.safetensors", "model.fp32-00002-of-00002.safetensors"], resume_download=True)

snapshot_download(repo_id="openai/whisper-large-v2", local_dir = "cache/whisper-large-v2", ignore_patterns=["tf_model.h5", "flax_model.msgpack", "model.safetensors", "model.fp32-00001-of-00002.safetensors", "model.fp32-00002-of-00002.safetensors"], resume_download=True)
