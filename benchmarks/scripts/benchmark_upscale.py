import subprocess
import time
import threading
import sys
import psutil

def run_upscale(sd_cli_path, label="tuned", input_img="output_tuned.png", output_img="output_upscaled.png"):
    cmd = [
        sd_cli_path,
        "-M", "upscale",
        "-i", input_img,
        "--upscale-model", r"C:\Users\abhik\Downloads\RealESRGAN_x4plus_anime_6B.pth",
        "-t", "8",
        "-o", output_img
    ]
    
    print(f"=== Running Upscaler Benchmark [{label}] ===")
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
    
    for line in proc.stdout:
        print(line, end="", flush=True)
        
    proc.wait()
    stop_monitor = True
    t.join(timeout=1.0)
    total_time = time.time() - start_time
    
    peak_mb = peak_rss / (1024 * 1024)
    peak_gb = peak_rss / (1024 * 1024 * 1024)
    
    print("\n" + "="*50)
    print(f"[{label} Upscale] COMPLETED in {total_time:.2f}s")
    print(f"[{label} Upscale] Peak WorkingSet64: {peak_mb:.1f} MB ({peak_gb:.2f} GB)")
    print("="*50 + "\n")
    return total_time, peak_mb, peak_gb

if __name__ == "__main__":
    cli = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\abhik\Documents\antigravity\silly-pascal\stable-diffusion.cpp\build\bin\sd-cli-tuned.exe"
    lbl = sys.argv[2] if len(sys.argv) > 2 else "tuned"
    out = sys.argv[3] if len(sys.argv) > 3 else "output_upscaled.png"
    run_upscale(cli, lbl, "output_tuned.png", out)
