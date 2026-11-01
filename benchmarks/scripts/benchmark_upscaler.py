import subprocess
import time
import threading
import sys
import psutil

def run_upscaler_benchmark(sd_cli_path, model_path, input_img="output_tuned.png", output_img="output_upscaled.png", tile_size=192):
    cmd = [
        sd_cli_path,
        "-M", "upscale",
        "--upscale-model", model_path,
        "-i", input_img,
        "-o", output_img,
        "--upscale-tile-size", str(tile_size),
        "-t", "8"
    ]
    
    print(f"=== Running Upscaler Benchmark ===")
    print(f"Model: {model_path}")
    print(f"Input: {input_img} (768x768) -> Output: {output_img} (3072x3072)")
    print(f"Tile Size: {tile_size}x{tile_size}")
    print("Command:", " ".join(cmd))
    
    start_time = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    peak_rss = 0
    stop_monitor = False
    
    def monitor_mem():
        nonlocal peak_rss
        try:
            p = psutil.Process(proc.pid)
            while not stop_monitor:
                try:
                    rss = p.memory_info().rss
                    if rss > peak_rss:
                        peak_rss = rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(0.05)
        except Exception:
            pass

    t = threading.Thread(target=monitor_mem, daemon=True)
    t.start()
    
    logs = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        logs.append(line)
        
    proc.wait()
    stop_monitor = True
    t.join(timeout=1.0)
    total_time = time.time() - start_time
    
    peak_mb = peak_rss / (1024 * 1024)
    peak_gb = peak_rss / (1024 * 1024 * 1024)
    
    print("\n" + "="*50)
    print(f"[UPSCALER] BENCHMARK COMPLETED in {total_time:.2f}s")
    print(f"[UPSCALER] Peak WorkingSet64: {peak_mb:.1f} MB ({peak_gb:.2f} GB)")
    print("="*50 + "\n")

if __name__ == "__main__":
    cli = r"C:\Users\abhik\Documents\antigravity\silly-pascal\stable-diffusion.cpp\build\bin\sd-cli-tuned.exe"
    model = r"C:\Users\abhik\Downloads\RealESRGAN_x4plus_anime_6B.pth"
    run_upscaler_benchmark(cli, model)
