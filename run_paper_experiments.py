"""Run the paper's main and reviewer evaluations, skipping completed artifacts.

Run this file again after an interruption. No checkpoints or results are bundled
with the repository; the publication pipeline downloads public datasets and
trains missing classifiers when requested below.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace

from revision.core import atomic_json, completed, mark_complete
from revision.publication import (
    ATTACKS,
    PublicationConfig,
    _load_verifier,
    dataset_object,
    detector_splits,
    load_classifier,
    resolve_device,
    resource_adjusted_config,
    run_verifier_attacks,
    safe_cuda_cleanup,
    train_classifier,
    train_signature_guidance,
    train_verifier,
)


def data_for(cfg):
    train = dataset_object(cfg.dataset, True, False)
    test = dataset_object(cfg.dataset, False, False)
    splits = detector_splits(train, cfg.seed, cfg.n_classes)
    return train, test, splits


def primary(device_name):
    command = [
        sys.executable, "run_publication.py", "--datasets", "cifar10,svhn,stl10",
        "--architectures", "resnet18", "--seeds", "0,1,2",
        "--train-missing-classifiers", "--device", device_name,
    ]
    print("Running primary publication matrix:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def controls(device):
    for dataset in ("cifar10", "svhn"):
        for seed in (0, 1, 2):
            cfg = resource_adjusted_config(replace(PublicationConfig(), dataset=dataset, seed=seed))
            output = cfg.run_dir / "ablations/no_signature/reviewer_matrix.json"
            attack_root = output.parent / "attacks"
            if completed(output) and all(completed(attack_root / f"{name}.json") for name in ATTACKS):
                print("Skipping completed control:", dataset, seed, flush=True)
                continue
            model, _ = load_classifier(cfg, device)
            train, test, splits = data_for(cfg)
            verifier, checkpoint = train_verifier(cfg, model, None, train, splits, device, "no_signature")
            attacks, clean = run_verifier_attacks(
                cfg, model, verifier, train, test, splits, device, False, ATTACKS, "no_signature"
            )
            atomic_json(
                {"dataset": dataset, "seed": seed, "variant": "no_signature",
                 "clean": clean, "attacks": attacks, "verifier_epoch": checkpoint.get("epoch")},
                output,
            )
            mark_complete(output)
            print("Completed control:", dataset, seed, flush=True)
            del model, verifier, train, test
            safe_cuda_cleanup()


def crossnorm(device):
    for dataset in ("cifar10", "svhn"):
        for variant in ("full", "no_signature"):
            cfg = resource_adjusted_config(replace(
                PublicationConfig(), dataset=dataset, seed=0, epsilon_l2=3.0, step_l2=0.3
            ))
            attack_dir = cfg.run_dir / ("attacks" if variant == "full" else "ablations/no_signature/attacks")
            output = attack_dir / "joint_pgd_l2.json"
            if completed(output):
                print("Skipping completed cross-norm attack:", dataset, variant, flush=True)
                continue
            model, _ = load_classifier(cfg, device)
            verifier_path = cfg.run_dir / ("verifier" if variant == "full" else "ablations/no_signature") / "verifier.pt"
            verifier, _ = _load_verifier(model, verifier_path, device)
            train, test, splits = data_for(cfg)
            run_verifier_attacks(
                cfg, model, verifier, train, test, splits, device, False, ("joint_pgd_l2",), variant
            )
            print("Completed cross-norm attack:", dataset, variant, flush=True)
            del model, verifier, train, test
            safe_cuda_cleanup()


def architecture(device):
    cfg = resource_adjusted_config(replace(
        PublicationConfig(), dataset="cifar10", architecture="densenet121", seed=0
    ))
    if not completed(cfg.classifier_checkpoint):
        train_classifier(cfg, device)
    model, _ = load_classifier(cfg, device)
    train, test, splits = data_for(cfg)
    _, geometry = train_signature_guidance(cfg, model, train, splits, device)
    verifier, _ = train_verifier(cfg, model, geometry, train, splits, device, "full")
    attacks = {}
    clean = None
    for attack in ("joint_pgd_linf", "joint_pgd_l2"):
        attack_cfg = replace(cfg, epsilon_l2=3.0, step_l2=0.3) if attack == "joint_pgd_l2" else cfg
        rows, clean = run_verifier_attacks(
            attack_cfg, model, verifier, train, test, splits, device, False, (attack,), "full"
        )
        attacks.update(rows)
    full_output = cfg.run_dir / "reviewer_architecture_summary.json"
    if not completed(full_output):
        atomic_json({"dataset": "cifar10", "architecture": "densenet121", "seed": 0,
                     "attacks": attacks, "clean": clean}, full_output)
        mark_complete(full_output)
    del verifier
    safe_cuda_cleanup()
    control, _ = train_verifier(cfg, model, None, train, splits, device, "no_signature")
    control_cfg = replace(cfg, final_attack_per_target=20)
    rows, control_clean = run_verifier_attacks(
        control_cfg, model, control, train, test, splits, device, False,
        ("joint_pgd_linf",), "no_signature"
    )
    control_output = cfg.run_dir / "reviewer_densenet_control.json"
    if not completed(control_output):
        atomic_json({"dataset": "cifar10", "architecture": "densenet121", "seed": 0,
                     "variant": "no_signature", "attacks": rows, "clean": control_clean,
                     "evaluation_scope": "20 successful targeted examples per class (200 total)"},
                    control_output)
        mark_complete(control_output)
    del model, control, train, test
    safe_cuda_cleanup()
    paired_output = cfg.run_dir / "reviewer_densenet_paired_pgd_200.json"
    if not completed(paired_output):
        subprocess.run([sys.executable, "summarize_densenet_paired_pilot.py", "--device", str(device)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "primary", "controls", "crossnorm", "architecture"), default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    args = parser.parse_args()
    if args.stage in ("all", "primary"):
        primary(args.device)
    if args.stage != "primary":
        device = resolve_device(args.device)
        if args.stage in ("all", "controls"):
            controls(device)
        if args.stage in ("all", "crossnorm"):
            crossnorm(device)
        if args.stage in ("all", "architecture"):
            architecture(device)


if __name__ == "__main__":
    main()
