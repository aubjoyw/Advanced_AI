import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from scipy import linalg
from tqdm import tqdm
import sys
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Inception feature extractor
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(torch.nn.Module):
    """
    Wraps InceptionV3 and returns the 2048-d pool features used by FID.
    We strip the classification head and hook the final average pool layer.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()

        # Remove the classifier — we only want pool features
        self.features = torch.nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, torch.nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        # Inception expects (B, 3, 299, 299) in [-1, 1] or [0, 1]
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.features(x)
        return x.squeeze(-1).squeeze(-1)   # (B, 2048)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_features_from_tensor(extractor, images, batch_size=64):
    """
    Extract Inception features from a (N, C, H, W) tensor in [-1, 1].
    Returns numpy array of shape (N, 2048).
    """
    dataset = TensorDataset(images)
    loader  = DataLoader(dataset, batch_size=batch_size)
    feats   = []
    for (batch,) in tqdm(loader, desc="Extracting features", leave=False,
                         dynamic_ncols=True, file=sys.stderr):
        feats.append(extractor(batch.to(extractor.device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


class ImageFolderDataset(Dataset):
    """Minimal dataset that loads images from a folder (no class subfolders needed)."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, folder, transform=None):
        self.paths = [
            p for p in Path(folder).rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found in {folder}")
        self.transform = transform or transforms.Compose([
            transforms.Resize((96, 96)),  # match your training resolution
            transforms.Resize((299, 299)),  # then resize to Inception input
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def extract_features_from_folder(extractor, folder, batch_size=64):
    """
    Extract Inception features from all images in a folder.
    Returns numpy array of shape (N, 2048).
    """
    dataset = ImageFolderDataset(folder)
    loader  = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    feats   = []
    for batch in tqdm(loader, desc=f"Extracting features from {folder}",
                      leave=False, dynamic_ncols=True, file=sys.stderr):
        feats.append(extractor(batch.to(extractor.device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_statistics(features):
    """Compute mean and covariance of a (N, 2048) feature array."""
    mu  = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mu, cov


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    Compute FID between two Gaussian distributions N(mu1, cov1) and N(mu2, cov2).

    FID = ||mu1 - mu2||² + Tr(cov1 + cov2 - 2 * sqrt(cov1 @ cov2))

    Lower is better. Typical ranges:
        < 10   excellent
        10–50  good
        50–100 moderate
        > 100  poor
    """
    diff = mu1 - mu2
    # Matrix square root via eigendecomposition (more stable than scipy sqrtm alone)
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)

    # sqrtm can produce tiny imaginary parts due to floating point — discard them
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            raise ValueError("Matrix square root has large imaginary component.")
        covmean = np.real(covmean)

    fid = diff @ diff + np.trace(cov1 + cov2 - 2 * covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def compute_fid_from_tensors(real_images, generated_images, batch_size=64, device="cuda"):
    """
    Compute FID between two sets of images provided as tensors.

    Args:
        real_images:      Tensor (N, 3, H, W) in [-1, 1] — your training images.
        generated_images: Tensor (N, 3, H, W) in [-1, 1] — model outputs from sample().
        batch_size:       Batch size for Inception forward passes.
        device:           "cuda" or "cpu".

    Returns:
        FID score (float). Lower is better.

    Example:
        from fid import compute_fid_from_tensors
        from diffusion import sample

        generated = sample(model, schedule, img_shape=(1000, 3, 64, 64))
        fid = compute_fid_from_tensors(real_batch, generated)
        print(f"FID: {fid:.2f}")
    """
    print("Loading Inception model...")
    extractor = InceptionFeatureExtractor(device=device)

    print("Extracting features from real images...")
    real_feats = extract_features_from_tensor(extractor, real_images, batch_size)

    print("Extracting features from generated images...")
    gen_feats  = extract_features_from_tensor(extractor, generated_images, batch_size)

    mu_r, cov_r = compute_statistics(real_feats)
    mu_g, cov_g = compute_statistics(gen_feats)

    fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    return fid


def compute_fid_from_folders(real_folder, generated_folder, batch_size=64, device="cuda"):
    """
    Compute FID between two folders of images.

    Args:
        real_folder:      Path to folder of real images (jpg/png, any structure).
        generated_folder: Path to folder of generated images.
        batch_size:       Batch size for Inception forward passes.
        device:           "cuda" or "cpu".

    Returns:
        FID score (float). Lower is better.

    Example:
        from fid import compute_fid_from_folders
        fid = compute_fid_from_folders("data/real", "outputs/generated")
        print(f"FID: {fid:.2f}")
    """
    print("Loading Inception model...")
    extractor = InceptionFeatureExtractor(device=device)

    print(f"Extracting features from real folder:      {real_folder}")
    real_feats = extract_features_from_folder(extractor, real_folder, batch_size)

    print(f"Extracting features from generated folder: {generated_folder}")
    gen_feats  = extract_features_from_folder(extractor, generated_folder, batch_size)

    mu_r, cov_r = compute_statistics(real_feats)
    mu_g, cov_g = compute_statistics(gen_feats)

    fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    return fid


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute FID between two image folders.")
    parser.add_argument("real",      help="Folder of real images")
    parser.add_argument("generated", help="Folder of generated images")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    fid = compute_fid_from_folders(
        args.real, args.generated,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"\nFID score: {fid:.4f}")
    print("(lower is better — <10 excellent, 10–50 good, 50–100 moderate, >100 poor)")
