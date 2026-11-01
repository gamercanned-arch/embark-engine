import subprocess
import time
import threading
import sys
import os
import psutil

def run_benchmark(sd_cli_path, label="stock", output_img="output.png"):
    cmd = [
        sd_cli_path,
        "-M", "img_gen",
        "--diffusion-model", r"C:\Users\abhik\Downloads\flux-2-klein-4b-Q3_K_M.gguf",
        "--llm", r"C:\Users\abhik\Downloads\qwen3-4b-abl-q4_0.gguf",
        "--vae", r"C:\Users\abhik\Downloads\flux2-vae.safetensors",
        "-p", "a cozy mountain cabin in a snowy pine forest at twilight, warm glowing windows, cinematic lighting, photorealistic",
        "-W", "768",
        "-H", "768",
        "--steps", "4",
        "--cfg-scale", "1.0",
        "--vae-tiling",
        "-t", "8",
        "-o", output_img
    ]
    
    print(f"=== Running Benchmark [{label}] ===")
    print("Command:", " ".join(cmd))
    
    start_time = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    peak_rss = 0
    rss_samples = []
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
                    rss_samples.append((time.time() - start_time, rss))
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
    print(f"[{label}] BENCHMARK COMPLETED in {total_time:.2f}s")
    print(f"[{label}] Peak WorkingSet64: {peak_mb:.1f} MB ({peak_gb:.2f} GB)")
    print("="*50 + "\n")
    
    return {
        "label": label,
        "total_time": total_time,
        "peak_rss_mb": peak_mb,
        "peak_rss_gb": peak_gb,
        "logs": logs
    }

if __name__ == "__main__":
    cli_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\abhik\Documents\antigravity\silly-pascal\stable-diffusion.cpp\build\bin\sd-cli.exe"
    label = sys.argv[2] if len(sys.argv) > 2 else "stock"
    output = sys.argv[3] if len(sys.argv) > 3 else f"output_{label}.png"
    run_benchmark(cli_path, label, output)
