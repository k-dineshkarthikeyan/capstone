from datasets import load_dataset
from huggingface_hub import login

login(token="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")


dataset = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "en",
    num_proc=20,
    # cache_dir="/mnt/e/huggingface/",
    trust_remote_code=True,
)
print("downloaded dataset")
