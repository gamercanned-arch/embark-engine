import urllib.request
import os
import time
import sys

URL = "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q3_K_M.gguf"
DST = r"C:\Users\abhik\Downloads\flux-2-klein-4b-Q3_K_M.gguf"
TOTAL_SIZE = 2124489280

def download():
    while True:
        existing = 0
        if os.path.exists(DST):
            existing = os.path.getsize(DST)
        
        if existing >= TOTAL_SIZE:
            print(f"Download complete: {existing} / {TOTAL_SIZE} bytes.")
            break
        
        print(f"Resuming from byte {existing} ({existing / (1024*1024):.2f} MB / {TOTAL_SIZE / (1024*1024):.2f} MB)...")
        req = urllib.request.Request(URL, headers={
            "User-Agent": "Mozilla/5.0",
            "Range": f"bytes={existing}-"
        })
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(DST, "ab") as f:
                start_t = time.time()
                bytes_batch = 0
                while True:
                    chunk = resp.read(1024 * 512) # 512 KB
                    if not chunk:
                        break
                    f.write(chunk)
                    existing += len(chunk)
                    bytes_batch += len(chunk)
                    elapsed = time.time() - start_t
                    if elapsed >= 5:
                        speed = bytes_batch / elapsed / (1024*1024)
                        percent = (existing / TOTAL_SIZE) * 100
                        print(f"Progress: {existing / (1024*1024):.1f} MB ({percent:.1f}%) - Speed: {speed:.2f} MB/s", flush=True)
                        start_t = time.time()
                        bytes_batch = 0
        except Exception as e:
            print(f"Connection dropped/retrying: {e}", flush=True)
            time.sleep(2)

if __name__ == "__main__":
    download()
