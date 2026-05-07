import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import random
import sys
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init
colorama_init()
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

from model import UNet
from diffusion import DiffusionSchedule, sample
from fid import (InceptionFeatureExtractor, extract_features_from_folder,
                 extract_features_from_tensor, compute_statistics, frechet_distance)


# --- Config ---
CHECKPOINT  = "checkpoints-pets/diffusion_epoch_500.pt"
NUM_IMAGES  = 9
IMG_SIZE    = 96
DIM_SIZE    = 64
OUTPUT_FILE = "generated.png"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# --- Sampler ---
SAMPLER    = "ddim"
DDIM_STEPS = 150
ETA        = 0.1

# --- Conditioning ---
# RANDOM_CLASSES: how many classes to randomly pick each run.
#   Set to None to use ALL classes every time.
#   Set to 0 for unconditional generation.
RANDOM_CLASSES = 9
GUIDANCE_SCALE = 3

# --- FID ---
COMPUTE_FID    = False
REAL_IMAGE_DIR = "/pet_images"

# --- Label rendering ---
FONT_SIZE = 5
LABEL_BG  = (30, 30, 30)
LABEL_FG  = (255, 255, 255)

# --- Generation batch size ---
# Lower this if you get CUDA OOM (try 16 or 8)
GEN_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def strip_compile_prefix(state_dict):
    prefix = "_orig_mod."
    return {(k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()}


def load_checkpoint(path, device=DEVICE):
    checkpoint  = torch.load(path, map_location=device)
    num_classes = checkpoint.get("num_classes", None)
    model       = UNet(img_channels=3, base_dim=DIM_SIZE, time_emb_dim=128,
                       num_classes=num_classes).to(device)

    def _is_noise(sd):
        key = "init_conv.weight"
        return key in sd and sd[key].float().std().item() < 0.01

    if "ema_state_dict" in checkpoint:
        ema_sd = strip_compile_prefix(checkpoint["ema_state_dict"])
        if not _is_noise(ema_sd):
            model.load_state_dict(ema_sd)
            weights_source = "EMA"
        else:
            print("Warning: EMA weights look untrained — falling back to raw weights.")
            model.load_state_dict(strip_compile_prefix(checkpoint["model_state_dict"]))
            weights_source = "raw training (EMA was bad)"
    else:
        model.load_state_dict(strip_compile_prefix(checkpoint["model_state_dict"]))
        weights_source = "raw training"

    model.eval()

    saved_names = checkpoint.get("class_names", None)
    class_names = ({i: name for i, name in enumerate(saved_names)}
                   if saved_names else {i: str(i) for i in range(num_classes or 0)})

    print(f"Loaded checkpoint  epoch={checkpoint['epoch']}  "
          f"loss={checkpoint['loss']:.4f}  "
          f"weights={weights_source}  "
          f"classes={num_classes if num_classes else 'unconditional'}")
    return model, class_names


# ---------------------------------------------------------------------------
# Class selection
# ---------------------------------------------------------------------------

def pick_classes(class_names, n):
    all_indices = list(class_names.keys())
    if n is None:
        return all_indices
    if n == 0:
        return None
    chosen = sorted(random.sample(all_indices, min(n, len(all_indices))))
    return chosen


def build_labels(classes, num_images, device):
    if classes is None:
        return None
    reps   = (num_images + len(classes) - 1) // len(classes)
    labels = (classes * reps)[:num_images]
    return torch.tensor(labels, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_batched(model, schedule, labels, num_images):
    """Generate num_images in batches of GEN_BATCH_SIZE to avoid CUDA OOM."""
    all_images = []
    num_batches = (num_images + GEN_BATCH_SIZE - 1) // GEN_BATCH_SIZE

    bar = tqdm(
        range(num_batches),
        desc=f"{Fore.BLUE}Generating",
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
        bar_format="{l_bar}" + Fore.BLUE + "{bar}" + "{r_bar}" + Style.RESET_ALL,
    )

    for i in bar:
        start = i * GEN_BATCH_SIZE
        end = min(start + GEN_BATCH_SIZE, num_images)
        batch_size = end - start
        batch_labels = labels[start:end] if labels is not None else None

        batch = sample(
            model, schedule,
            img_shape=(batch_size, 3, IMG_SIZE, IMG_SIZE),
            sampler=SAMPLER,
            ddim_steps=DDIM_STEPS,
            eta=ETA,
            labels=batch_labels,
            guidance_scale=GUIDANCE_SCALE,
            device=DEVICE,
            show_progress=False,  # suppress inner DDIM bar so outer bar stays in place
        )
        all_images.append(batch.cpu())
        torch.cuda.empty_cache()

    return torch.cat(all_images, dim=0)


# ---------------------------------------------------------------------------
# Label rendering
# ---------------------------------------------------------------------------

def _load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except IOError:
        return ImageFont.load_default()


def _measure(text, font):
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    return tmp.textbbox((0, 0), text, font=font)


def _wrap_text(text, font, max_w):
    words, lines, current = text.split(), [], ""
    for word in words:
        test = (current + " " + word).strip()
        if _measure(test, font)[2] <= max_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text]


def add_label_bar(img_tensor, label_text, label_height, font_size):
    pil_img   = TF.to_pil_image(img_tensor)
    W, H      = pil_img.size
    font      = _load_font(font_size)
    max_width = W - 4

    lines  = _wrap_text(label_text, font, max_width)
    line_h = _measure("A", font)[3] + 2
    text_h = line_h * len(lines)

    canvas = Image.new("RGB", (W, H + label_height), LABEL_BG)
    canvas.paste(pil_img, (0, 0))
    draw   = ImageDraw.Draw(canvas)

    top = H + (label_height - text_h) // 2
    for i, line in enumerate(lines):
        bbox   = _measure(line, font)
        text_w = bbox[2] - bbox[0]
        draw.text(((W - text_w) // 2, top + i * line_h), line, fill=LABEL_FG, font=font)

    return TF.to_tensor(canvas)


def save_labelled_grid(images, labels, class_names, output_file, nrow, font_size=FONT_SIZE):
    if labels is None:
        from torchvision.utils import save_image
        save_image(images, output_file, nrow=nrow)
        return

    font      = _load_font(font_size)
    max_width = IMG_SIZE - 4

    # Pass 1 — find tallest label bar needed
    names, max_bar_h = [], 0
    for i in range(len(images)):
        name = class_names.get(labels[i].item(), str(labels[i].item()))
        names.append(name)
        line_h   = _measure("A", font)[3] + 2
        needed_h = line_h * len(_wrap_text(name, font, max_width)) + 4
        max_bar_h = max(max_bar_h, needed_h)

    # Pass 2 — render with uniform bar height
    labelled = [add_label_bar(img, names[i], max_bar_h, font_size)
                for i, img in enumerate(images)]

    TF.to_pil_image(make_grid(labelled, nrow=nrow, padding=2)).save(output_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    schedule           = DiffusionSchedule(timesteps=1000, schedule="cosine", device=DEVICE)
    model, CLASS_NAMES = load_checkpoint(CHECKPOINT)

    # When computing FID use all classes for a fair comparison
    n_classes = None if COMPUTE_FID else RANDOM_CLASSES
    classes   = pick_classes(CLASS_NAMES, n_classes)
    labels    = build_labels(classes, NUM_IMAGES, DEVICE)

    if labels is not None and not COMPUTE_FID:
        chosen_names = [CLASS_NAMES[c] for c in classes]
        print(f"Classes: {chosen_names}  (guidance scale: {GUIDANCE_SCALE})")
    else:
        print("Generating unconditionally.")

    if NUM_IMAGES < 500 and COMPUTE_FID:
        print(f"Warning: NUM_IMAGES={NUM_IMAGES} is low — FID is unreliable below ~500.")

    # --- Generate (always batched) ---
    generated = generate_batched(model, schedule, labels, NUM_IMAGES)

    # --- Save grid ---
    if not COMPUTE_FID:
        images_01 = (generated + 1) / 2
        nrow      = int(NUM_IMAGES ** 0.5)
        save_labelled_grid(images_01, labels, CLASS_NAMES, OUTPUT_FILE,
                           nrow=nrow, font_size=FONT_SIZE)
        print(f"Saved -> {OUTPUT_FILE}")

    # --- FID ---
    if COMPUTE_FID:
        print(f"\nComputing FID against: {REAL_IMAGE_DIR}")
        extractor   = InceptionFeatureExtractor(device=DEVICE)
        real_feats  = extract_features_from_folder(extractor, REAL_IMAGE_DIR)
        gen_feats   = extract_features_from_tensor(extractor, generated)
        mu_r, cov_r = compute_statistics(real_feats)
        mu_g, cov_g = compute_statistics(gen_feats)
        fid         = frechet_distance(mu_r, cov_r, mu_g, cov_g)
        print(f"\nFID score: {fid:.4f}")
        print("(lower is better — <10 excellent, 10–50 good, 50–100 moderate, >100 poor)")
