# src/embeddings_facenet.py
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FaceNetEmbedder:
    """
    Dùng FaceNet (InceptionResnetV1 pretrained='vggface2') để trích 512-d embedding.
    Ảnh đầu vào nên đã align/crop mặt. Resize về 160x160 và normalize [-1,1].
    """
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(_DEVICE)
        self.tf = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1]
        ])

    @torch.no_grad()
    def embed_pil(self, img: Image.Image) -> np.ndarray:
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(_DEVICE)
        vec = self.model(x).cpu().numpy()[0].astype("float32")
        return vec

def load_image_any(path_str: str, project_root: Path | None = None) -> Image.Image:
    p = Path(path_str)
    cands = [p]
    if project_root is not None:
        cands += [project_root / p, project_root / Path(str(path_str).replace("\\", "/"))]
    for c in cands:
        if c.exists():
            return Image.open(c).convert("RGB")
    raise FileNotFoundError(path_str)
