"""production_vlm.robustness: adversarial/natural perturbation testing, OOD detection, hallucination guarding.

Implements P1-02 of the Production VLM Engineering roadmap: a
robustness and safety layer for multimodal vision systems, covering
the three concrete failure modes called out in the original plan --
adversarial/natural perturbation robustness, embedding-space
out-of-distribution inputs, and ungrounded (hallucinated) generated
text -- plus the "wrap VLM inference with guard-style checks"
integration pattern tying them together.

Modules:
    perturbations   Natural (ImageNet-C-style) and gradient-based (PGD) adversarial perturbations.
    ood             Per-sample k-NN out-of-distribution detection in embedding space.
    guard           Hallucination/grounding guard: turns eval metrics into pass/flag/reject decisions.
"""

from production_vlm.robustness.guard import GuardConfig, GuardDecision, GuardResult, HallucinationGuard
from production_vlm.robustness.ood import KNNOODDetector, OODResult
from production_vlm.robustness.perturbations import (
    NaturalPerturbation,
    PerturbationResult,
    apply_perturbation,
    pgd_attack,
)

__all__ = [
    "GuardConfig",
    "GuardDecision",
    "GuardResult",
    "HallucinationGuard",
    "KNNOODDetector",
    "OODResult",
    "NaturalPerturbation",
    "PerturbationResult",
    "apply_perturbation",
    "pgd_attack",
]
