# Benchmark DINOv2-small su CPU

Eseguito il 7 novembre 2026 su VM `XBRASILE` (Windows 11 Enterprise
multi-session, build 26200), CPU virtuale QEMU 8 core/logical processor, 31,93 GB
RAM totali (22,41 GB disponibili al controllo), Python 3.11.15, PyTorch
2.15.0+cpu, 4 thread Torch. Modello ufficiale `facebook/dinov2-small`, pesi
Safetensors dalla cache locale del progetto. Benchmark `benchmark.py`; immagini
RGB sintetiche uniformi 392x392 e 518x518, processor senza resize/crop.

| Input | Patch | Embedding (incluso CLS) | Inferenza dopo caricamento | RSS dopo modello | Picco RSS campionato |
|---|---:|---|---:|---:|---:|
| 392 px | 28x28 | 1x785x384 | 0,911 / 0,805 s | 310,4 MB | 448,3 MB |
| 518 px | 37x37 | 1x1370x384 | 1,771 / 1,591 s | 310,4 MB | 448,3 MB |

Seconda inferenza tipica: 0,805 s a 392 e 1,591 s a 518. I tempi includono
preprocessing immagine; escludono caricamento pesi e processor. La RSS è del
processo Python; picco campionato ogni 20 ms durante i due test. Risultati:
device `cpu`, `torch.cuda.is_available() == False`, embedding tutti finiti,
norma media patch 58,81 (392) e 59,32 (518). La PoC resta non validata per
accuratezza su corrosione reale.
