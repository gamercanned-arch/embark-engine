import requests
import threading
import time
import json

BASE_URL = "http://127.0.0.1:8080/v1/chat/completions"

def send_chat(model_name, session_id, results):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": f"Session {session_id} ping"}],
        "stream": True,
        "max_tokens": 15
    }
    t_send = time.time()
    first_chunk_t = None
    chunks = []
    try:
        r = requests.post(BASE_URL, json=payload, stream=True, timeout=10)
        for line in r.iter_lines():
            if line:
                if first_chunk_t is None:
                    first_chunk_t = time.time()
                chunks.append(line.decode('utf-8'))
        t_done = time.time()
        results.append({
            "id": session_id,
            "model": model_name,
            "send_t": t_send,
            "first_chunk_t": first_chunk_t,
            "done_t": t_done,
            "chunks": len(chunks)
        })
    except Exception as e:
        results.append({
            "id": session_id,
            "model": model_name,
            "error": str(e)
        })

print("[*] Launching 2 Concurrent Groot-19B Requests simultaneously...")
results_groot = []
t0 = time.time()
t_a = threading.Thread(target=send_chat, args=("groot-19b", "Groot_A", results_groot))
t_b = threading.Thread(target=send_chat, args=("groot-19b", "Groot_B", results_groot))
t_a.start()
t_b.start()
t_a.join()
t_b.join()

print("\n--- Groot Serialization Verification ---")
for r in sorted(results_groot, key=lambda x: x["first_chunk_t"]):
    print(f"[{r['id']}] First chunk at: {r['first_chunk_t']-t0:.3f}s | Completed at: {r['done_t']-t0:.3f}s | Chunks: {r['chunks']}")

r1, r2 = sorted(results_groot, key=lambda x: x["first_chunk_t"])
# Strict serial check: Second session only starts receiving AFTER first session finishes!
is_strict_serial = r2["first_chunk_t"] >= r1["done_t"] - 0.05
print(f"\n[+] Strict Serial Execution: {'CONFIRMED (Second request waited for first to finish)' if is_strict_serial else 'FAILED'}")
