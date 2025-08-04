import os
from huggingface_hub import hf_hub_download

def download_weights():
    repo_id = "Burf/DrUM"
    model_dir = "./weight"
    
    files = ["L.safetensors", "H.safetensors", "bigG.safetensors", "T5.safetensors"]
    
    os.makedirs(model_dir, exist_ok = True)
    for filename in files:
        filepath = os.path.join(model_dir, filename)
        
        if os.path.exists(filepath):
            print("✅ {0} - already exist".format(filename))
            continue
            
        print("📥 {0} - downloading...".format(filename))
        try:
            hf_hub_download(
                repo_id = repo_id,
                filename = "weight/" + filename,
                local_dir = ".",
                local_dir_use_symlinks = False
            )
            print("✅ {0} - success".format(filename))
        except Exception as e:
            print("❌ {0} - fail: {1}".format(filename, e))
    
    print("🎉 Finish!")

if __name__ == "__main__":
    download_weights()