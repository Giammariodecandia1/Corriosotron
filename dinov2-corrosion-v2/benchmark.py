"""Benchmark CPU reale per DINOv2-small; richiede pesi gia' in cache locale."""
import json
import os
import threading
import time

import psutil
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "facebook/dinov2-small"
THREADS = min(4, os.cpu_count() or 1)
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)
assert not torch.cuda.is_available(), "Benchmark previsto esclusivamente su CPU"

proc = psutil.Process()
processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_ID, use_safetensors=True, local_files_only=True).to("cpu").eval()
baseline = proc.memory_info().rss
peak = [baseline]
stop = threading.Event()

def sample_memory():
    while not stop.is_set():
        try:
            peak[0] = max(peak[0], proc.memory_info().rss)
        except psutil.Error:
            pass
        stop.wait(0.02)

sampler = threading.Thread(target=sample_memory, daemon=True)
sampler.start()
results = []
try:
    for size in (392, 518):
        image = Image.new("RGB", (size, size), (128, 128, 128))
        inputs = processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        durations = []
        output = None
        for _ in range(2):
            start = time.perf_counter()
            with torch.inference_mode():
                output = model(**inputs).last_hidden_state
            durations.append(time.perf_counter() - start)
            peak[0] = max(peak[0], proc.memory_info().rss)
        patches = output[:, 1:, :]
        results.append({
            "input_px": size,
            "grid": [size // 14, size // 14],
            "embedding_shape_including_cls": list(output.shape),
            "patch_embedding_shape": list(patches.shape),
            "all_finite": bool(torch.isfinite(output).all()),
            "mean_patch_norm": float(patches.norm(dim=-1).mean()),
            "inference_seconds_after_load": [round(x, 3) for x in durations],
            "rss_after_inference_mb": round(proc.memory_info().rss / 1048576, 1),
        })
finally:
    stop.set()
    sampler.join()

print(json.dumps({
    "model": MODEL_ID,
    "device": str(next(model.parameters()).device),
    "cuda_available": torch.cuda.is_available(),
    "torch_version": torch.__version__,
    "cpu_threads": THREADS,
    "process_rss_after_model_load_mb": round(baseline / 1048576, 1),
    "process_peak_rss_sampled_mb": round(peak[0] / 1048576, 1),
    "sampling_interval_ms": 20,
    "results": results,
}, indent=2))
