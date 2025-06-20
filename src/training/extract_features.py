import os
import sys
import torch
import torchvision.transforms as T
import cv2
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dino_lora import LoRADINO

def load_image(path, image_size=224):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found: {path}")
    img = img[:, :, ::-1]  # BGR to RGB
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0)

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LoRADINO().to(device)
    model.eval()

    image_paths = glob("data/*.png") + glob("data/*.jpg")
    os.makedirs("features", exist_ok=True)

    for path in image_paths:
        img = load_image(path).to(device)
        with torch.no_grad():
            features = model(img)              # [1, 257, 768]
            patch_features = features[:, 1:]   # [1, 256, 768] → exclude [CLS]

        fname = os.path.splitext(os.path.basename(path))[0]
        torch.save(patch_features.cpu(), f"features/{fname}.pt")
        print(f"✅ Saved features for {fname} with shape {patch_features.shape}")
