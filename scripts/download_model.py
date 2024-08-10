# scripts/download_model.py
import requests

def download_model():
    url = "https://your-cloud-storage.com/best_model_yolo.pt"
    local_filename = "models/best_model_yolo.pt"
    
    print(f"Downloading {local_filename} from {url}...")
    response = requests.get(url, stream=True)
    
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Download complete: {local_filename}")

if __name__ == "__main__":
    download_model()
