# scripts/download_model.py
import requests

def download_model():
    url = 'https://drive.google.com/uc?export=download&id=1Gk8Wx5eT8dr2g_b8dw1MIlSRptGzsKk'
    local_filename = "models/best_model_yolo.pt"
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(32768):
                f.write(chunk)
        print(f"Model downloaded successfully to {local_filename}")
    else:
        print(f"Failed to download file, status code {response.status_code}")

if __name__ == "__main__":
    download_model()
