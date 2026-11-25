import os
from pathlib import Path
os.environ["HF_HOME"]=str(Path(__file__).resolve().parent/"model_cache")
from main import model
from PIL import Image
import torch
processor,m=model()
with torch.inference_mode():
    x=processor(images=Image.new("RGB",(392,392)),return_tensors="pt",do_resize=False,do_center_crop=False)
    y=m(**x).last_hidden_state
print("DINOv2 reale caricato su CPU:",tuple(y.shape))
