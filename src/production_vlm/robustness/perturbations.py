"""Adversarial and natural perturbation generators for vision robustness testing.

Two families of perturbation are implemented, matching the two kinds
of robustness failure that matter in production CV/VLM systems:

1. **Natural/distributional perturbations** (`NaturalPerturbation`) --
   lighting, contrast, noise, blur, occlusion, rotation. These are the
   perturbations a deployed system actually encounters (a slightly
   darker camera feed, a smudged lens, a rotated document scan) and
   are the right default robustness check for most teams, since they
   require no access to model gradients and still catch a large
   fraction of real-world brittleness.

2. **Gradient-based adversarial attacks** (`pgd_attack`) -- a
   PGD (Projected Gradient Descent) implementation following Madry et
   al., "Towards Deep Learning Models Resistant to Adversarial
   Attacks" (2018), the standard reference attack for evaluating
   L-infinity-bounded adversarial robustness. This requires a real
   differentiable model (PyTorch) and is therefore gated behind the
   `ml` extra; without it, `pgd_attack` raises a clear `ImportError`
   rather than silently no-op'ing.

Both families operate on numpy arrays in [0, 1], HWC or NHWC layout,
so they compose cleanly with `production_vlm.utils.synthetic_charts` and
with real image pipelines alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class PerturbationResult:
    name: str
    severity: float
    perturbed_image: Image.Image
    params: dict


class NaturalPerturbation:
    """Library of natural/distributional perturbations applied to PIL images.

    Each method takes a severity in [0, 1] (0 = no change, 1 = maximum
    configured strength) so a robustness sweep can report accuracy as
    a function of severity rather than a single pass/fail point --
    the same convention used by ImageNet-C (Hendrycks & Dietterich,
    2019) for benchmarking corruption robustness.
    """

    @staticmethod
    def brightness(image: Image.Image, severity: float) -> PerturbationResult:
        """Darkens the image proportionally to severity (simulates low-light / underexposed capture)."""
        severity = float(np.clip(severity, 0.0, 1.0))
        arr = np.asarray(image, dtype=np.float64) / 255.0
        delta = -severity * 0.6  # up to 60% darker at severity=1
        arr = np.clip(arr * (1.0 + delta), 0.0, 1.0)
        out = Image.fromarray((arr * 255).astype(np.uint8))
        return PerturbationResult("brightness", severity, out, {"delta": float(delta)})

    @staticmethod
    def contrast(image: Image.Image, severity: float) -> PerturbationResult:
        severity = float(np.clip(severity, 0.0, 1.0))
        arr = np.asarray(image, dtype=np.float64) / 255.0
        mean = arr.mean()
        factor = 1.0 - severity * 0.7  # up to 70% contrast reduction at severity=1
        arr = np.clip((arr - mean) * factor + mean, 0.0, 1.0)
        out = Image.fromarray((arr * 255).astype(np.uint8))
        return PerturbationResult("contrast", severity, out, {"factor": float(factor)})

    @staticmethod
    def gaussian_noise(image: Image.Image, severity: float, seed: int = 0) -> PerturbationResult:
        severity = float(np.clip(severity, 0.0, 1.0))
        rng = np.random.default_rng(seed)
        arr = np.asarray(image, dtype=np.float64) / 255.0
        sigma = severity * 0.25  # up to std=0.25 in [0,1] pixel space at severity=1
        noisy = np.clip(arr + rng.normal(0, sigma, size=arr.shape), 0.0, 1.0)
        out = Image.fromarray((noisy * 255).astype(np.uint8))
        return PerturbationResult("gaussian_noise", severity, out, {"sigma": float(sigma)})

    @staticmethod
    def gaussian_blur(image: Image.Image, severity: float) -> PerturbationResult:
        severity = float(np.clip(severity, 0.0, 1.0))
        radius = severity * 6.0  # up to radius=6px at severity=1
        out = image.filter(ImageFilter.GaussianBlur(radius=radius)) if radius > 0 else image.copy()
        return PerturbationResult("gaussian_blur", severity, out, {"radius": float(radius)})

    @staticmethod
    def rotation(image: Image.Image, severity: float) -> PerturbationResult:
        severity = float(np.clip(severity, 0.0, 1.0))
        angle = severity * 25.0  # up to 25 degrees at severity=1 -- realistic for a skewed document scan
        out = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        return PerturbationResult("rotation", severity, out, {"angle_degrees": float(angle)})

    @staticmethod
    def occlusion(image: Image.Image, severity: float, seed: int = 0) -> PerturbationResult:
        """Paints a random opaque rectangle over part of the image, simulating a sticker/finger/glare occlusion."""
        severity = float(np.clip(severity, 0.0, 1.0))
        rng = np.random.default_rng(seed)
        arr = np.asarray(image).copy()
        h, w = arr.shape[:2]
        box_frac = severity * 0.4  # up to 40% of width/height at severity=1
        box_w, box_h = int(w * box_frac), int(h * box_frac)
        if box_w > 0 and box_h > 0:
            x0 = rng.integers(0, max(1, w - box_w))
            y0 = rng.integers(0, max(1, h - box_h))
            arr[y0 : y0 + box_h, x0 : x0 + box_w] = 128  # mid-gray occluder
        out = Image.fromarray(arr)
        return PerturbationResult("occlusion", severity, out, {"box_frac": box_frac})

    ALL: dict[str, Callable[[Image.Image, float], PerturbationResult]] = {}


NaturalPerturbation.ALL = {
    "brightness": NaturalPerturbation.brightness,
    "contrast": NaturalPerturbation.contrast,
    "gaussian_noise": NaturalPerturbation.gaussian_noise,
    "gaussian_blur": NaturalPerturbation.gaussian_blur,
    "rotation": NaturalPerturbation.rotation,
    "occlusion": NaturalPerturbation.occlusion,
}


def apply_perturbation(image: Image.Image, kind: str, severity: float, seed: int = 0) -> PerturbationResult:
    """Dispatch to one of `NaturalPerturbation.ALL` by name."""
    if kind not in NaturalPerturbation.ALL:
        raise ValueError(f"Unknown perturbation kind '{kind}'. Available: {sorted(NaturalPerturbation.ALL)}")
    fn = NaturalPerturbation.ALL[kind]
    try:
        return fn(image, severity, seed=seed)  # type: ignore[call-arg]
    except TypeError:
        return fn(image, severity)  # type: ignore[call-arg]


def pgd_attack(
    model: "object",
    image_tensor: "object",
    target_loss_fn: Callable,
    epsilon: float = 8 / 255,
    alpha: float = 2 / 255,
    n_steps: int = 10,
):
    """L-infinity-bounded Projected Gradient Descent attack (Madry et al., 2018).

    Requires PyTorch and a real differentiable `model` -- this is the
    genuine gradient-based attack, not a numpy approximation, since a
    PGD attack without real gradients would be misleading rather than
    merely simplified. Gated behind the `ml` extra:

        pip install -e ".[ml]"

    Args:
        model: a `torch.nn.Module` mapping image tensors to logits/embeddings.
        image_tensor: a `torch.Tensor` of shape (1, C, H, W) in [0, 1], requires_grad not needed (set internally).
        target_loss_fn: callable(model_output) -> scalar torch.Tensor loss to *maximize* (the attack ascends this).
        epsilon: L-infinity perturbation budget (default 8/255, the standard CIFAR/ImageNet attack budget).
        alpha: per-step gradient ascent size (default 2/255, following Madry et al.'s convention of epsilon/4).
        n_steps: number of PGD iterations (default 10, the standard "PGD-10" evaluation setting).

    Returns:
        (adversarial_image_tensor, perturbation_linf_norm) as torch tensors.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "pgd_attack requires PyTorch (the `ml` extra: pip install -e \".[ml]\"). "
            "A numpy-only approximation would not be a real gradient-based attack, "
            "so this raises rather than silently substituting a weaker check."
        ) from e

    original = image_tensor.clone().detach()
    perturbed = original.clone().detach()
    # Random start within the epsilon-ball, per Madry et al.'s recommendation
    # for stronger attacks than a zero-init start.
    perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0, 1).detach()

    for _ in range(n_steps):
        perturbed.requires_grad_(True)
        output = model(perturbed)
        loss = target_loss_fn(output)
        grad = torch.autograd.grad(loss, perturbed)[0]

        perturbed = perturbed.detach() + alpha * grad.sign()
        delta = torch.clamp(perturbed - original, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(original + delta, 0, 1).detach()

    linf_norm = float((perturbed - original).abs().max().item())
    return perturbed, linf_norm
