from huggingface_hub import hf_hub_download, hf_hub_cache
import shutil

cache_dir = hf_hub_cache()
shutil.rmtree(cache_dir)
