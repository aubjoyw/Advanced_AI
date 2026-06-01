import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
import time
import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from tqdm import tqdm
import sys
from colorama import Fore, Style, init as colorama_init
colorama_init()

from model import UNet
from diffusion import DiffusionSchedule
from plot_loss import plot_loss

#NeuralNine

# --- Config ---
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE       = 96
BATCH_SIZE     = 16
DIM_SIZE       = 64
EPOCHS         = 600
LR             = 2e-4
SAVE_EVERY     = 50
CHECKPOINT_DIR = "Final/checkpoints-pets" # Change to checkpoint directory
DATA_DIR       = "Final/pet_images"   # Change to your dataset path

# --- Conditioning ---
# Set NUM_CLASSES to the number of classes in your dataset.
# Set to None to train unconditionally (no class labels).
NUM_CLASSES     = 37    # e.g. 10 for CIFAR-10; None for unconditional
CFG_DROP_PROB   = 0.10  # probability of dropping the label during training

# --- EMA ---
EMA_DECAY      = 0.9999  # how slowly the EMA tracks the training model
                          # 0.9999 = very slow/stable (good for long runs)
                          # 0.999  = faster tracking (better for short runs)
EMA_WARMUP     = 2000    # number of steps before EMA starts updating
                          # keeps early noisy weights from polluting the average


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """
    Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMA(model, decay=0.9999, warmup=2000)

        # After each optimizer step:
        ema.update(model, step)

        # To generate with EMA weights:
        with ema.apply(model):
            images = sample(model, ...)
        # model weights are automatically restored afterwards
    """

    def __init__(self, model, decay=0.9999, warmup=2000):
        self.decay = decay
        self.warmup = warmup
        # Always deepcopy the underlying uncompiled module so the shadow
        # has clean parameter names without the _orig_mod. prefix
        raw = getattr(model, "_orig_mod", model)
        # Store a deep copy of the initial weights as the EMA shadow
        self.shadow = copy.deepcopy(raw).eval()
        # EMA model never needs gradients
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _effective_decay(self, step):
        """
        Ramp the decay up from 0 during warmup so early noisy weights
        don't get locked into the EMA average.
        """
        return min(self.decay, (1 + step) / (self.warmup + step))

    @torch.no_grad()
    def update(self, model, step):
        decay = self._effective_decay(step)
        # Always pull from the underlying uncompiled module — torch.compile wraps
        # parameters and can cause silent device/shape mismatches in the zip
        raw = getattr(model, "_orig_mod", model)
        # Ensure shadow is on the same device as the model
        shadow_device = next(self.shadow.parameters()).device
        model_device = next(raw.parameters()).device
        if shadow_device != model_device:
            self.shadow = self.shadow.to(model_device)
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
        # Also update buffers (e.g. BatchNorm running stats)
        for ema_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def apply(self, model):
        """
        Context manager that temporarily swaps model weights with EMA weights.
        Original weights are restored on exit — safe to use mid-training.

        Example:
            with ema.apply(model):
                out = model(x, t)   # runs with EMA weights
            out = model(x, t)       # back to training weights
        """
        return _EMAContext(self.shadow, model)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


class _EMAContext:
    """Temporarily swaps a model's parameters with the EMA shadow's."""
    def __init__(self, shadow, model):
        self.shadow = shadow
        self.model  = model
        self.backup = None

    def __enter__(self):
        # Save training weights, load EMA weights into the model
        self.backup = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow.state_dict())
        return self.model

    def __exit__(self, *_):
        # Restore training weights
        self.model.load_state_dict(self.backup)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def strip_compile_prefix(state_dict):
    """Strip _orig_mod. prefix added by torch.compile so checkpoints are portable."""
    prefix = "_orig_mod."
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def save_checkpoint(model, optimizer, ema, epoch, loss, all_loss, path=CHECKPOINT_DIR, class_names=None):
    os.makedirs(path, exist_ok=True)
    save_path = f"{path}/diffusion_epoch_{epoch}.pt"
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     strip_compile_prefix(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict":       strip_compile_prefix(ema.state_dict()),
        "loss":                 loss,
        "num_classes":          model.num_classes,
        "class_names":          class_names,  # saved so generate.py can read labels automatically
        "all_loss":             all_loss,
    }, save_path)
    #tqdm.write(f"  Checkpoint saved -> {save_path}")


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def get_dataloader():
    no_aug_transform = transforms.Compose([
        # --- To tensor and normalise ---
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # scale to [-1, 1],
    ])
    transform = transforms.Compose([
        # --- Geometry ---
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),  # mirror ~half the images
        transforms.RandomRotation(degrees=15),  # rotate up to ±15°
        transforms.RandomAffine(  # translate + slight shear
            degrees=0,
            translate=(0.1, 0.1),
            shear=5,
        ),
        transforms.RandomResizedCrop(  # zoom into a random crop
            size=IMG_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),

        # --- Colour ---
        transforms.ColorJitter(
            brightness=0.3,  # ± brightness
            contrast=0.3,  # ± contrast
            saturation=0.2,  # ± saturation
            hue=0.05,  # subtle hue shift
        ),
        transforms.RandomGrayscale(p=0.05),  # occasionally desaturate

        # --- Noise / blur ---
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.2,  # apply blur 20% of the time
        ),

        # --- To tensor and normalise ---
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # scale to [-1, 1],
    ])

    # dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform)
    # For your own images with class subfolders:
    orig_dataset = datasets.ImageFolder(DATA_DIR, transform=no_aug_transform)
    trans_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    trans_dataset2 = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_dataset = torch.utils.data.ConcatDataset([orig_dataset, trans_dataset, trans_dataset2])

    # Strip the ImageNet prefix (e.g. "n02085620-Chihuahua" -> "Chihuahua")
    # and replace underscores with spaces for readable labels
    class_names = [
        #name.split("-", 1)[-1].replace("_", " ")  # Standford Dogs
        name.replace("_", " ")                     # Oxford Pets
        for name in orig_dataset.classes
    ]

    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=4, pin_memory=True, persistent_workers=True,
                      prefetch_factor=2), class_names


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, schedule, dataloader, class_names=None, start_epoch=0, start_step=0, ema=None, allLoss = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    scaler    = GradScaler('cuda')
    model.to(DEVICE)

    if ema is None:
        ema = EMA(model, decay=EMA_DECAY, warmup=EMA_WARMUP)
    ema.shadow.to(DEVICE)

    step      = start_step
    epoch_bar = tqdm(range(start_epoch, EPOCHS), unit="epoch",
                     dynamic_ncols=True, file=sys.stderr,
                     desc=f"{Fore.BLUE}Training",
                     bar_format="{l_bar}" + Fore.BLUE + "{bar}" + "{r_bar}" + Style.RESET_ALL)

    if allLoss is not None:
        all_loss = allLoss
    else:
        all_loss = []

    start = time.time()
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        for x0, labels in dataloader:
            x0 = x0.to(DEVICE, memory_format=torch.channels_last)
            labels = labels.to(DEVICE) if NUM_CLASSES is not None else None
            t = torch.randint(0, schedule.T, (x0.shape[0],), device=DEVICE)

            if labels is not None:
                drop_mask = torch.rand(labels.shape[0], device=DEVICE) < CFG_DROP_PROB
                labels[drop_mask] = model.null_token

            with autocast('cuda'):
                x_t, noise = schedule.q_sample(x0, t)
                noise_pred = model(x_t, t, labels)
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA shadow after every optimizer step
            ema.update(model, step)
            step += 1
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        all_loss.append(avg_loss)
        # Show avg loss on the epoch bar
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        if (epoch + 1) % SAVE_EVERY == 0:
           # tqdm.write(f"  Epoch {epoch+1}: avg loss {avg_loss:.4f}")
            save_checkpoint(model, optimizer, ema, epoch + 1, avg_loss, all_loss, class_names=class_names)

        scheduler.step()

    save_checkpoint(model, optimizer, ema, EPOCHS, avg_loss, all_loss, class_names=class_names)
    tqdm.write("Training complete.")

    # Print training time
    total_time = time.time() - start
    total_days = int(total_time / 86400)
    total_time = total_time - total_days * 86400
    total_hours = int(total_time / 3600)
    total_time = total_time - total_hours * 3600
    total_minutes = int(total_time / 60)
    total_time = total_time - total_minutes * 60
    print(f"Total time: {total_days}:{total_hours}:{total_minutes}:{total_time:.2f}.")

    return all_loss


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def resume_training(checkpoint_path):
    checkpoint  = torch.load(checkpoint_path, map_location=DEVICE)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    model       = UNet(img_channels=3, base_dim=DIM_SIZE, time_emb_dim=128, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]

    dataloader, class_names = get_dataloader()          # one call only
    start_step = start_epoch * len(dataloader)          # now correct

    ema = EMA(model, decay=EMA_DECAY, warmup=EMA_WARMUP)
    if "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        tqdm.write("EMA weights restored from checkpoint.")

    tqdm.write(f"Resuming from epoch {start_epoch} (loss: {checkpoint['loss']:.4f})")
    schedule = DiffusionSchedule(timesteps=1000, schedule="cosine", device=DEVICE)
    loss = train(model, schedule, dataloader, class_names=class_names,
                 start_epoch=start_epoch, start_step=start_step, ema=ema, allLoss=checkpoint["all_loss"])

    return loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    schedule = DiffusionSchedule(timesteps=1000, schedule="cosine", device=DEVICE)
    dataloader, class_names = get_dataloader()
    num_classes = len(class_names)

    # Model creation
    model = UNet(img_channels=3, base_dim=DIM_SIZE, time_emb_dim=128, num_classes=num_classes)
    inductor_config.max_autotune_gemm = False   # Possibly delete if on Tempest
    model = torch.compile(model)
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Breeds found:     {num_classes}")
    print(f"EMA decay:        {EMA_DECAY}  warmup: {EMA_WARMUP} steps")
    loss = train(model, schedule, dataloader, class_names=class_names)
    # loss = resume_training("C:\\Users\\aubjo\\PycharmProjects\\Advanced_AI\\Final\\checkpoints-pets\\diffusion_epoch_500.pt")

    plot_loss(loss)
