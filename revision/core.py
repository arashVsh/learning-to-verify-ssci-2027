from __future__ import annotations

import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


FORCE_RERUN_COMPLETED = False


@dataclass(frozen=True)
class Config:
    dataset: str = "cifar10"
    seed: int = 0
    classifier_seed: int = 0
    stage: int = 1
    root: str = "Results/revision"
    train_per_class: int = 400
    test_per_class: int = 100
    calibration_per_class: int = 100
    attack_batch_size: int = 32
    attack_steps: int = 80
    epsilon: float = 8 / 255
    step_size: float = 2 / 255
    detector_lambda: float = 1.0
    eot_draws: int = 8
    full_individual: int = 25
    full_joint: int = 5
    light_individual: int = 20
    light_joint: int = 4
    aggressive_individual: int = 10
    aggressive_joint: int = 2
    alpha: float = 0.5
    n_last_layers: int = 8

    @property
    def run_dir(self) -> Path:
        return Path(self.root) / self.dataset / f"seed_{self.seed}"


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    os.close(fd)
    try:
        torch.save(value, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".partial", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def completed(path: Path, force: bool = False) -> bool:
    return path.exists() and path.with_suffix(path.suffix + ".complete").exists() and not force


def mark_complete(path: Path) -> None:
    marker = path.with_suffix(path.suffix + ".complete")
    marker.write_text("complete\n", encoding="ascii")


class NormalizedModel(torch.nn.Module):
    """Accept pixel-space [0,1] inputs; normalize exactly once before the classifier."""
    def __init__(self, model, mean, std):
        super().__init__(); self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, x): return self.model((x - self.mean) / self.std)


class GraphActivationExtractor:
    """Same layer-selection rule/order as Detection.ActivationExtractors, without detach."""
    def __init__(self, model, n_last_layers: int, params_threshold: int = 1000):
        self.activations, self.handles = [], []
        significant = []
        for name, module in reversed(list(model.named_modules())):
            if sum(p.numel() for p in module.parameters()) > params_threshold:
                significant.append((name, module))
                if len(significant) == n_last_layers: break
        self.layer_names = [n for n, _ in reversed(significant)]
        for _, module in reversed(significant):
            self.handles.append(module.register_forward_hook(self._hook))
    def _hook(self, module, inputs, output):
        if isinstance(output, tuple): output = output[0]
        if torch.is_tensor(output): self.activations.append(output)
    def reset(self): self.activations.clear()
    def matrix(self): return torch.cat([a.flatten(1) for a in self.activations], 1)
    def close(self):
        for h in self.handles: h.remove()
        self.handles.clear()


def choose_targets(labels: torch.Tensor, n_classes: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=labels.device).manual_seed(seed)
    offsets = torch.randint(1, n_classes, labels.shape, generator=g, device=labels.device)
    return (labels + offsets) % n_classes


@torch.no_grad()
def targeted_success(model, x, targets): return model(x).argmax(1).eq(targets)


def assert_targeted(model, x, targets, context="adversarial pool") -> None:
    ok = targeted_success(model, x, targets)
    assert bool(ok.all()), f"{context}: {(~ok).sum().item()} retained samples do not predict as their intended target"


class DetectorSurrogate(torch.nn.Module):
    """Class-conditioned approximation to predicted-class detector banks."""
    def __init__(self, in_features: int, n_classes: int = 1):
        super().__init__(); self.n_classes=n_classes
        # Non-persistent buffers keep legacy checkpoint compatibility. V3
        # artifacts store and restore these explicitly.
        self.register_buffer("feature_mean",torch.zeros(in_features),persistent=False)
        self.register_buffer("feature_scale",torch.ones(in_features),persistent=False)
        self.trunk=torch.nn.Sequential(torch.nn.Linear(in_features,256),torch.nn.SiLU(),torch.nn.Dropout(.1),
                                       torch.nn.Linear(256,96),torch.nn.SiLU())
        self.heads=torch.nn.ModuleList([torch.nn.Linear(96,1) for _ in range(n_classes)])
    def set_normalization(self,mean,scale):
        self.feature_mean.copy_(mean.to(self.feature_mean));self.feature_scale.copy_(scale.to(self.feature_scale).clamp_min(1e-4))
    def logits(self,x,classes=None):
        h=self.trunk((x-self.feature_mean)/self.feature_scale)
        if classes is None:
            if self.n_classes != 1: raise ValueError("Class-conditioned surrogate requires classes")
            return self.heads[0](h).squeeze(1)
        all_scores=torch.cat([head(h) for head in self.heads],1)
        return all_scores.gather(1,classes.long().view(-1,1)).squeeze(1)
    def forward(self,x,classes=None): return torch.sigmoid(self.logits(x,classes))
    def evasion_value(self,x,classes,use_logits=False):
        return self.logits(x,classes) if use_logits else self.forward(x,classes)


class CrossBankGeometry(torch.nn.Module):
    """Exact differentiable geometry over MI-selected neurons from every class bank."""
    def __init__(self,indices,centers,scales,distance_center,distance_scale,
                 margin_center,margin_scale,temperature=.25):
        super().__init__();self.temperature=float(temperature)
        self.register_buffer("indices",torch.as_tensor(indices,dtype=torch.long))
        self.register_buffer("centers",torch.as_tensor(centers,dtype=torch.float32))
        self.register_buffer("scales",torch.as_tensor(scales,dtype=torch.float32).clamp_min(1e-4))
        self.register_buffer("distance_center",torch.as_tensor(distance_center,dtype=torch.float32))
        self.register_buffer("distance_scale",torch.as_tensor(distance_scale,dtype=torch.float32).clamp_min(.05))
        self.register_buffer("margin_center",torch.as_tensor(margin_center,dtype=torch.float32))
        self.register_buffer("margin_scale",torch.as_tensor(margin_scale,dtype=torch.float32).clamp_min(.05))

    def distances(self,activations):
        selected=activations[:,self.indices]
        z=(selected-self.centers.unsqueeze(0))/self.scales.unsqueeze(0)
        return F.smooth_l1_loss(z,torch.zeros_like(z),beta=1.,reduction="none").mean(2)

    def components(self,activations,classes):
        distances=self.distances(activations);classes=classes.long();row=torch.arange(len(classes),device=classes.device)
        chosen=distances[row,classes]
        mask=F.one_hot(classes,num_classes=distances.shape[1]).bool()
        other=distances.masked_fill(mask,float("inf"))
        soft_other=-self.temperature*torch.logsumexp(-other/self.temperature,dim=1)
        margin=chosen-soft_other
        return distances,chosen,margin

    def forward(self,activations,classes):
        _,chosen,margin=self.components(activations,classes);classes=classes.long()
        distance_z=(chosen-self.distance_center[classes])/self.distance_scale[classes]
        margin_z=(margin-self.margin_center[classes])/self.margin_scale[classes]
        return (F.smooth_l1_loss(distance_z,torch.zeros_like(distance_z),reduction="none")+
                F.smooth_l1_loss(margin_z,torch.zeros_like(margin_z),reduction="none"))


def adaptive_geometry_pgd(model,extractor,geometry,images,targets,epsilon,step_size,steps,
                          detector_lambda=.2,random_start=True):
    """Targeted PGD with exact gradients through the reported cross-bank detector."""
    x0=images.detach();x=x0+torch.empty_like(x0).uniform_(-epsilon,epsilon) if random_start else x0.clone()
    x=x.clamp(0,1).detach()
    for _ in range(steps):
        x.requires_grad_(True);extractor.reset();logits=model(x);score=geometry(extractor.matrix(),targets)
        per=F.cross_entropy(logits,targets,reduction="none")+detector_lambda*score
        grad=torch.autograd.grad(per.sum(),x)[0]
        if not torch.isfinite(grad).all() or grad.abs().max()==0:raise RuntimeError("Cross-bank adaptive gradient invalid")
        x=x.detach()-step_size*grad.sign();x=torch.max(torch.min(x,x0+epsilon),x0-epsilon).clamp(0,1)
    return x.detach()


def adaptive_geometry_pgd_multistart(model,extractor,geometry,images,targets,epsilon,step_size,steps,
                                     detector_lambdas=(.05,.2,.8)):
    best=images.detach().clone();best_score=torch.full((len(images),),float("inf"),device=images.device)
    for detector_lambda in detector_lambdas:
        candidate=adaptive_geometry_pgd(model,extractor,geometry,images,targets,epsilon,step_size,steps,
                                        detector_lambda,True)
        with torch.no_grad():
            extractor.reset();pred=model(candidate).argmax(1);score=geometry(extractor.matrix(),targets)
            better=pred.eq(targets)&(score<best_score);best[better]=candidate[better];best_score[better]=score[better]
    return best.detach()


def adaptive_baseline_pgd(model,extractor,images,targets,epsilon,step_size,steps,kind,artifact,
                          detector_lambdas=(.05,.2,.8)):
    """Direct adaptive attack for energy, MSP, Mahalanobis, or activation-logistic baselines."""
    x0=images.detach();best=x0.clone();best_score=torch.full((len(images),),float("inf"),device=images.device)
    for detector_lambda in detector_lambdas:
        x=(x0+torch.empty_like(x0).uniform_(-epsilon,epsilon)).clamp(0,1).detach()
        for _ in range(steps):
            x.requires_grad_(True);extractor.reset();logits=model(x);acts=extractor.matrix();classes=targets.long()
            if kind=="energy":value=-torch.logsumexp(logits,1)
            elif kind=="msp":value=1-logits.softmax(1).max(1).values
            elif kind=="diagonal_mahalanobis":
                mean=artifact["means"].to(x)[classes];var=artifact["variances"].to(x)[classes]
                value=((acts-mean).pow(2)/var).mean(1)
            elif kind=="activation_logistic":
                idx=artifact["feature_idx"].to(x.device);coef=artifact["logistic_coef"].to(x);bias=artifact["logistic_bias"].to(x)
                # Optimize the unsaturated decision function; probability is
                # monotone in this logit but can have effectively zero gradient.
                value=acts[:,idx].matmul(coef)+bias
            else:raise ValueError(kind)
            scale_key="activation_logistic_decision_scale" if kind=="activation_logistic" else f"{kind}_scale"
            scale=artifact[scale_key].to(x)[classes].clamp_min(1e-4)
            detector=value/scale
            per=F.cross_entropy(logits,targets,reduction="none")+detector_lambda*detector
            grad=torch.autograd.grad(per.sum(),x)[0]
            if not torch.isfinite(grad).all() or grad.abs().max()==0:raise RuntimeError(f"{kind} adaptive gradient invalid")
            x=x.detach()-step_size*grad.sign();x=torch.max(torch.min(x,x0+epsilon),x0-epsilon).clamp(0,1)
        with torch.no_grad():
            extractor.reset();logits=model(x);acts=extractor.matrix();classes=targets.long()
            if kind=="energy":value=-torch.logsumexp(logits,1)
            elif kind=="msp":value=1-logits.softmax(1).max(1).values
            elif kind=="diagonal_mahalanobis":
                value=((acts-artifact["means"].to(x)[classes]).pow(2)/artifact["variances"].to(x)[classes]).mean(1)
            else:
                value=torch.sigmoid(acts[:,artifact["feature_idx"].to(x.device)].matmul(artifact["logistic_coef"].to(x))+artifact["logistic_bias"].to(x))
            score=value
            better=logits.argmax(1).eq(targets)&(score<best_score);best[better]=x[better];best_score[better]=score[better]
    return best.detach()


def adaptive_guard_pgd(model,guard,images,targets,epsilon,step_size,steps,guard_weights=(.25,1.,4.),
                       extractor=None,geometry=None,geometry_weight=0.):
    """Attack the classifier while explicitly forcing the guard to accept the target."""
    x0=images.detach();best=x0.clone();best_score=torch.full((len(images),),float("inf"),device=images.device)
    for guard_weight in guard_weights:
        x=(x0+torch.empty_like(x0).uniform_(-epsilon,epsilon)).clamp(0,1).detach()
        for _ in range(steps):
            x.requires_grad_(True)
            if extractor is not None:extractor.reset()
            logits=model(x);guard_logits=guard(x)
            per=F.cross_entropy(logits,targets,reduction="none")+guard_weight*F.cross_entropy(guard_logits,targets,reduction="none")
            if geometry is not None and geometry_weight>0:
                per=per+geometry_weight*geometry(extractor.matrix(),targets)
            grad=torch.autograd.grad(per.sum(),x)[0]
            if not torch.isfinite(grad).all() or grad.abs().max()==0:raise RuntimeError("Guard-adaptive gradient invalid")
            x=x.detach()-step_size*grad.sign();x=torch.max(torch.min(x,x0+epsilon),x0-epsilon).clamp(0,1)
        with torch.no_grad():
            logits=model(x);guard_prob=guard(x).softmax(1);score=1-guard_prob[torch.arange(len(x),device=x.device),targets]
            better=logits.argmax(1).eq(targets)&(score<best_score);best[better]=x[better];best_score[better]=score[better]
    return best.detach()


def verifier_training_attack(model,verifier,extractor,geometry,images,true_labels,targets,
                             epsilon,step_size,steps,verifier_weight=1.,geometry_weight=.05):
    """Create targeted, signature-camouflaged examples that maximally hurt the verifier."""
    x0=images.detach();x=(x0+torch.empty_like(x0).uniform_(-epsilon,epsilon)).clamp(0,1).detach()
    for _ in range(steps):
        x.requires_grad_(True)
        if geometry_weight>0:
            if extractor is None or geometry is None:raise ValueError("Signature geometry requires an activation extractor and geometry model")
            extractor.reset()
        base_logits=model(x);verifier_logits=verifier(x)
        # Minimize target CE for the fixed classifier, maximize verifier loss
        # on the true label, and enter the clean signature geometry.
        per=(F.cross_entropy(base_logits,targets,reduction="none")-
             verifier_weight*F.cross_entropy(verifier_logits,true_labels,reduction="none"))
        if geometry_weight>0:per=per+geometry_weight*geometry(extractor.matrix(),targets)
        grad=torch.autograd.grad(per.sum(),x)[0]
        if not torch.isfinite(grad).all() or grad.abs().max()==0:raise RuntimeError("Verifier-training gradient invalid")
        x=x.detach()-step_size*grad.sign();x=torch.max(torch.min(x,x0+epsilon),x0-epsilon).clamp(0,1)
    return x.detach()


def adaptive_verifier_pgd(model,verifier,images,targets,epsilon,step_size,steps,
                          verifier_weights=(.1,.5,2.,8.),initial=None):
    """Directly force both fixed classifier and robust verifier to the same target."""
    x0=images.detach();best=x0.clone();best_score=torch.full((len(images),),float("inf"),device=images.device)
    for weight in verifier_weights:
        if initial is None:x=(x0+torch.empty_like(x0).uniform_(-epsilon,epsilon)).clamp(0,1).detach()
        else:
            delta=(initial.detach()-x0).clamp(-epsilon,epsilon)
            x=(x0+delta).clamp(0,1).detach()
        for _ in range(steps):
            x.requires_grad_(True);base_logits=model(x);verifier_logits=verifier(x)
            per=(F.cross_entropy(base_logits,targets,reduction="none")+
                 weight*F.cross_entropy(verifier_logits,targets,reduction="none"))
            grad=torch.autograd.grad(per.sum(),x)[0]
            if not torch.isfinite(grad).all() or grad.abs().max()==0:raise RuntimeError("Verifier-adaptive gradient invalid")
            x=x.detach()-step_size*grad.sign();x=torch.max(torch.min(x,x0+epsilon),x0-epsilon).clamp(0,1)
        with torch.no_grad():
            base_logits=model(x);prob=verifier(x).softmax(1);score=1-prob[torch.arange(len(x),device=x.device),targets]
            better=base_logits.argmax(1).eq(targets)&(score<best_score);best[better]=x[better];best_score[better]=score[better]
    return best.detach()


def adaptive_verifier_apgd(model,verifier,images,targets,epsilon,steps,
                           verifier_weights=(.1,.5,2.,8.)):
    """Targeted joint APGD-style attack with per-example best tracking."""
    x0=images.detach();global_best=x0.clone();global_score=torch.full((len(x0),),float("inf"),device=x0.device)
    checkpoints={max(1,int(steps*q)) for q in (.22,.4,.55,.7,.85)}
    for weight in verifier_weights:
        x=(x0+torch.empty_like(x0).uniform_(-epsilon,epsilon)).clamp(0,1);best=x.clone()
        best_loss=torch.full((len(x),),float("inf"),device=x.device);step=torch.full((len(x),1,1,1),2*epsilon,device=x.device)
        for i in range(steps):
            x.requires_grad_(True);base=model(x);verify=verifier(x)
            per=F.cross_entropy(base,targets,reduction="none")+weight*F.cross_entropy(verify,targets,reduction="none")
            improved=per.detach()<best_loss;best_loss[improved]=per.detach()[improved];best[improved]=x.detach()[improved]
            grad=torch.autograd.grad(per.sum(),x)[0]
            if not torch.isfinite(grad).all() or grad.abs().max()==0:raise RuntimeError("Verifier APGD gradient invalid")
            x=x.detach()-step*grad.sign();x=torch.max(torch.min(x,x0+epsilon),x0-epsilon).clamp(0,1)
            if i+1 in checkpoints:step*=.5;x=best.clone()
        with torch.no_grad():
            base=model(best);prob=verifier(best).softmax(1);score=1-prob[torch.arange(len(best),device=best.device),targets]
            better=base.argmax(1).eq(targets)&(score<global_score);global_best[better]=best[better];global_score[better]=score[better]
    return global_best.detach()


def adaptive_verifier_l2_pgd(model,verifier,images,targets,epsilon,step_size,steps,
                             verifier_weights=(.1,.5,2.,8.)):
    """Targeted joint PGD constrained to an L2 ball."""
    x0=images.detach();global_best=x0.clone();global_score=torch.full((len(x0),),float("inf"),device=x0.device)
    for weight in verifier_weights:
        delta=torch.randn_like(x0);flat=delta.flatten(1);delta=delta/(flat.norm(2,1).view(-1,1,1,1)+1e-12)
        radius=torch.rand(len(x0),1,1,1,device=x0.device)*epsilon;x=(x0+delta*radius).clamp(0,1).detach()
        for _ in range(steps):
            x.requires_grad_(True);base=model(x);verify=verifier(x)
            per=F.cross_entropy(base,targets,reduction="none")+weight*F.cross_entropy(verify,targets,reduction="none")
            grad=torch.autograd.grad(per.sum(),x)[0];norm=grad.flatten(1).norm(2,1).view(-1,1,1,1)
            if not torch.isfinite(grad).all() or norm.max()==0:raise RuntimeError("Verifier L2-PGD gradient invalid")
            x=x.detach()-step_size*grad/(norm+1e-12);delta=x-x0
            dnorm=delta.flatten(1).norm(2,1).view(-1,1,1,1);delta=delta*torch.minimum(torch.ones_like(dnorm),epsilon/(dnorm+1e-12))
            x=(x0+delta).clamp(0,1)
        with torch.no_grad():
            base=model(x);prob=verifier(x).softmax(1);score=1-prob[torch.arange(len(x),device=x.device),targets]
            better=base.argmax(1).eq(targets)&(score<global_score);global_best[better]=x[better];global_score[better]=score[better]
    return global_best.detach()


def adaptive_verifier_cw(model,verifier,images,targets,steps,max_l2=3.,learning_rate=1e-2,
                         verifier_weights=(.25,1.,4.)):
    """Targeted CW-L2 attack jointly optimizing classifier and verifier margins."""
    x0=images.detach();eps=1e-6;global_best=x0.clone();global_score=torch.full((len(x0),),float("inf"),device=x0.device)
    for weight in verifier_weights:
        w=torch.atanh((x0*2-1).clamp(-1+eps,1-eps)).detach().requires_grad_(True);opt=torch.optim.Adam([w],lr=learning_rate)
        best=x0.clone();best_score=torch.full((len(x0),),float("inf"),device=x0.device)
        for _ in range(steps):
            x=(torch.tanh(w)+1)/2;base=model(x);verify=verifier(x);row=torch.arange(len(x),device=x.device)
            other_base=base.masked_fill(F.one_hot(targets,base.shape[1]).bool(),-1e9).max(1).values
            other_verify=verify.masked_fill(F.one_hot(targets,verify.shape[1]).bool(),-1e9).max(1).values
            margin=(other_base-base[row,targets]).clamp_min(0)+weight*(other_verify-verify[row,targets]).clamp_min(0)
            l2=(x-x0).flatten(1).pow(2).sum(1);loss=(l2+10*margin).sum();opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                score=1-verify.softmax(1)[row,targets];ok=base.argmax(1).eq(targets)&(l2.sqrt()<=max_l2)
                better=ok&(score<best_score);best[better]=x[better];best_score[better]=score[better]
        better=best_score<global_score;global_best[better]=best[better];global_score[better]=best_score[better]
    return global_best.detach()


def adaptive_pgd(model, extractor, surrogate, images, targets, epsilon, step_size,
                 steps, detector_lambda, random_start=True, progress=None,
                 use_surrogate_logits=False, initial=None):
    """Minimize target CE plus adversarial score; success is prediction==preselected target."""
    x0 = images.detach()
    if initial is not None: x=initial.detach().clone()
    else: x = x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon) if random_start else x0.clone()
    x = x.clamp(0, 1).detach()
    for i in range(steps):
        x.requires_grad_(True); extractor.reset(); logits = model(x)
        score = surrogate.evasion_value(extractor.matrix(),targets,use_surrogate_logits)
        loss = F.cross_entropy(logits, targets) + detector_lambda * score.mean()
        grad = torch.autograd.grad(loss, x)[0]
        if not torch.isfinite(grad).all() or grad.abs().max() == 0:
            raise RuntimeError("Adaptive detector gradient is zero or non-finite")
        x = x.detach() - step_size * grad.sign()
        x = torch.max(torch.min(x, x0 + epsilon), x0 - epsilon).clamp(0, 1)
        if progress and (i == 0 or (i + 1) % 10 == 0): progress(i + 1, steps)
    return x.detach()


def adaptive_apgd(model, extractor, surrogate, images, targets, epsilon, steps, detector_lambda,
                  use_surrogate_logits=False):
    """Targeted APGD-style attack with adaptive step halving and the same detector objective."""
    x0=images.detach(); x=(x0+torch.empty_like(x0).uniform_(-epsilon,epsilon)).clamp(0,1)
    best=x.clone(); best_loss=torch.full((len(x),),float("inf"),device=x.device); step=2*epsilon
    checkpoints=set([max(1,int(steps*q)) for q in (.22,.4,.55,.7,.85)])
    for i in range(steps):
        x.requires_grad_(True);extractor.reset();logits=model(x);score=surrogate.evasion_value(extractor.matrix(),targets,use_surrogate_logits)
        per=F.cross_entropy(logits,targets,reduction="none")+detector_lambda*score
        improved=per.detach()<best_loss;best_loss[improved]=per.detach()[improved];best[improved]=x.detach()[improved]
        grad=torch.autograd.grad(per.sum(),x)[0]
        if not torch.isfinite(grad).all() or grad.abs().max()==0: raise RuntimeError("DA-APGD gradient invalid")
        x=(x.detach()-step*grad.sign());x=torch.max(torch.min(x,x0+epsilon),x0-epsilon).clamp(0,1)
        if i+1 in checkpoints: step*=.5;x=best.clone()
    return x.detach()


def adaptive_cw(model, extractor, surrogate, images, targets, steps, detector_lambda,
                learning_rate=5e-3, max_l2=3.0,use_surrogate_logits=False):
    """Targeted CW-L2 objective with a directly connected detector-surrogate term."""
    eps=1e-6;x0=images.detach();w=torch.atanh((x0*2-1).clamp(-1+eps,1-eps)).detach().requires_grad_(True)
    opt=torch.optim.Adam([w],lr=learning_rate);best=x0.clone();best_l2=torch.full((len(x0),),float("inf"),device=x0.device)
    for _ in range(steps):
        x=(torch.tanh(w)+1)/2;extractor.reset();logits=model(x);score=surrogate.evasion_value(extractor.matrix(),targets,use_surrogate_logits)
        other=logits.clone();other[torch.arange(len(x)),targets]=-1e9
        margin=(other.max(1).values-logits[torch.arange(len(x)),targets]).clamp_min(0)
        l2=(x-x0).flatten(1).pow(2).sum(1);loss=(l2+10*margin+detector_lambda*score).sum()
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            ok=model(x).argmax(1).eq(targets)&(l2.sqrt()<=max_l2);better=ok&(l2<best_l2)
            best[better]=x[better];best_l2[better]=l2[better]
    return best.detach()


def ordinary_targeted_pgd(model, images, targets, epsilon, step_size, steps):
    x0 = images.detach(); x = (x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)).clamp(0, 1)
    for _ in range(steps):
        x.requires_grad_(True); loss = F.cross_entropy(model(x), targets)
        grad = torch.autograd.grad(loss, x)[0]
        x = x.detach() - step_size * grad.sign()
        x = torch.max(torch.min(x, x0 + epsilon), x0 - epsilon).clamp(0, 1)
    return x.detach()


def collect_successful(attack: Callable, model, loader: Iterable, n_required: int,
                       n_classes: int, seed: int, device, label: str):
    xs, ys, ts, attempted = [], [], [], 0; started = time.time()
    for batch_i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device); t = choose_targets(y, n_classes, seed + attempted)
        adv = attack(x, t); ok = targeted_success(model, adv, t); attempted += len(x)
        if ok.any():
            xs.append(adv[ok].cpu()); ys.append(y[ok].cpu()); ts.append(t[ok].cpu())
        got = sum(len(z) for z in xs); elapsed = time.time() - started
        print(f"[{time.strftime('%H:%M:%S')}][{label}] batch={batch_i} attempted={attempted} "
              f"target_success={got}/{n_required} elapsed={elapsed/60:.1f}m", flush=True)
        if got >= n_required: break
    if not xs: raise RuntimeError(f"{label}: no successful targeted examples")
    adv, true, targets = torch.cat(xs)[:n_required], torch.cat(ys)[:n_required], torch.cat(ts)[:n_required]
    assert_targeted(model, adv.to(device), targets.to(device), label)
    return {"adversarial": adv, "true_labels": true, "targets": targets,
            "attempted": attempted, "successful": sum(len(z) for z in xs),
            "retained": len(adv), "targeted_asr": sum(len(z) for z in xs) / attempted}


def collect_successful_balanced(attack: Callable,model,loader,per_target,n_classes,seed,device,label):
    """Collect exactly `per_target` successful examples for every predetermined target."""
    buckets={c:{"x":[],"original":[],"y":[],"t":[]} for c in range(n_classes)};attempted=successful=0;started=time.time()
    for batch_i,(x,y) in enumerate(loader):
        x,y=x.to(device),y.to(device)
        # Deterministic preselection independent of attack outcome, cycling target offsets.
        offsets=(torch.arange(attempted,attempted+len(y),device=device)+seed)%(n_classes-1)+1
        t=(y+offsets)%n_classes;adv=attack(x,t);ok=targeted_success(model,adv,t);attempted+=len(x);successful+=int(ok.sum())
        for cls in range(n_classes):
            room=per_target-sum(len(z) for z in buckets[cls]["x"])
            take=torch.where(ok&t.eq(cls))[0][:max(0,room)]
            if len(take):
                buckets[cls]["x"].append(adv[take].cpu());buckets[cls]["y"].append(y[take].cpu());buckets[cls]["t"].append(t[take].cpu())
                buckets[cls]["original"].append(x[take].detach().cpu())
        counts=[sum(len(z) for z in buckets[c]["x"]) for c in range(n_classes)]
        print(f"[{time.strftime('%H:%M:%S')}][{label}] batch={batch_i} attempted={attempted} successful={successful} "
              f"per_target={min(counts)}/{per_target} total={sum(counts)}/{per_target*n_classes} elapsed={(time.time()-started)/60:.1f}m",flush=True)
        if min(counts)>=per_target: break
    counts=[sum(len(z) for z in buckets[c]["x"]) for c in range(n_classes)]
    if min(counts)<per_target: raise RuntimeError(f"{label}: exhausted candidates; per-target counts={counts}")
    adv=torch.cat([torch.cat(buckets[c]["x"])[:per_target] for c in range(n_classes)])
    true=torch.cat([torch.cat(buckets[c]["y"])[:per_target] for c in range(n_classes)])
    targets=torch.cat([torch.cat(buckets[c]["t"])[:per_target] for c in range(n_classes)])
    originals=torch.cat([torch.cat(buckets[c]["original"])[:per_target] for c in range(n_classes)])
    assert_targeted(model,adv.to(device),targets.to(device),label)
    return {"adversarial":adv,"original":originals,"true_labels":true,"targets":targets,"attempted":attempted,
            "successful":successful,"retained":len(adv),"targeted_asr":successful/attempted,
            "retained_per_target":{str(c):per_target for c in range(n_classes)}}


def metrics(clean_scores, adv_scores, threshold=None):
    clean_scores, adv_scores = np.asarray(clean_scores), np.asarray(adv_scores)
    y = np.r_[np.zeros(len(clean_scores)), np.ones(len(adv_scores))]
    s = np.r_[clean_scores, adv_scores]
    if threshold is None: threshold = float(np.quantile(clean_scores, .95, method="higher"))
    fpr, tpr, _ = roc_curve(y, s)
    at = lambda q: float(tpr[np.where(fpr <= q)[0][-1]]) if np.any(fpr <= q) else 0.0
    return {"roc_auc": float(roc_auc_score(y, s)), "tpr_at_1pct_fpr": at(.01),
            "tpr_at_5pct_fpr": at(.05), "threshold": threshold,
            "clean_fpr": float((clean_scores >= threshold).mean()),
            "detector_bypass": float((adv_scores < threshold).mean()),
            "mean_clean_score": float(clean_scores.mean()), "mean_adversarial_score": float(adv_scores.mean())}


def class_conditional_thresholds(clean_scores, predicted, fpr=.05, n_classes=10):
    clean_scores=np.asarray(clean_scores);predicted=np.asarray(predicted);global_t=float(np.quantile(clean_scores,1-fpr,method="higher"))
    thresholds={}
    for cls in range(n_classes):
        vals=clean_scores[predicted==cls]
        thresholds[cls]=float(np.quantile(vals,1-fpr,method="higher")) if len(vals)>=20 else global_t
    return thresholds


def metrics_class_conditional(clean_scores,clean_pred,adv_scores,adv_pred,thresholds):
    result=metrics(clean_scores,adv_scores,threshold=float(np.mean(list(thresholds.values()))))
    clean_t=np.asarray([thresholds[int(c)] for c in clean_pred]);adv_t=np.asarray([thresholds[int(c)] for c in adv_pred])
    result["threshold"]={str(k):float(v) for k,v in thresholds.items()}
    result["clean_fpr"]=float((np.asarray(clean_scores)>=clean_t).mean())
    result["detector_bypass"]=float((np.asarray(adv_scores)<adv_t).mean())
    return result


def fit_class_conditional_reference(values, predicted, n_classes=10):
    """Store sorted clean scores for finite-sample, class-conditional typicality."""
    values=np.asarray(values,dtype=np.float64);predicted=np.asarray(predicted,dtype=np.int64)
    if len(values)!=len(predicted) or not np.isfinite(values).all():
        raise ValueError("Typicality reference requires aligned finite values and predictions")
    global_values=np.sort(values)
    result={}
    for cls in range(n_classes):
        class_values=np.sort(values[predicted==cls])
        result[cls]=class_values if len(class_values)>=10 else global_values
    return result


def empirical_two_sided_p(values, predicted, reference):
    """Smoothed two-sided empirical p-values; both tails are treated as atypical."""
    values=np.asarray(values,dtype=np.float64);predicted=np.asarray(predicted,dtype=np.int64)
    result=np.empty(len(values),dtype=np.float64)
    for cls in np.unique(predicted):
        idx=np.where(predicted==cls)[0];ref=np.asarray(reference[int(cls)],dtype=np.float64);n=len(ref)
        if n==0:raise ValueError(f"Empty typicality reference for class {cls}")
        lower=np.searchsorted(ref,values[idx],side="left")
        upper=n-np.searchsorted(ref,values[idx],side="right")
        result[idx]=np.minimum(1.,2.*np.minimum(lower+1,upper+1)/(n+1.))
    return result


def empirical_tail_p(values,predicted,reference,tail):
    """Smoothed empirical p-value for a prespecified lower or upper tail."""
    if tail not in ("lower","upper"):raise ValueError("tail must be 'lower' or 'upper'")
    values=np.asarray(values,dtype=np.float64);predicted=np.asarray(predicted,dtype=np.int64)
    result=np.empty(len(values),dtype=np.float64)
    for cls in np.unique(predicted):
        idx=np.where(predicted==cls)[0];ref=np.asarray(reference[int(cls)],dtype=np.float64);n=len(ref)
        if n==0:raise ValueError(f"Empty typicality reference for class {cls}")
        count=np.searchsorted(ref,values[idx],side="right") if tail=="lower" else n-np.searchsorted(ref,values[idx],side="left")
        result[idx]=(count+1)/(n+1.)
    return result


def fisher_typicality_score(*p_values):
    """Fisher-style evidence fusion, used only with an empirically calibrated threshold."""
    if not p_values:raise ValueError("At least one p-value vector is required")
    arrays=[np.asarray(p,dtype=np.float64) for p in p_values]
    if any(len(p)!=len(arrays[0]) for p in arrays):raise ValueError("p-value vectors must align")
    return -2.*sum(np.log(np.clip(p,1e-12,1.)) for p in arrays)


def adaptive_pgd_multistart(model,extractor,surrogate,images,targets,epsilon,step_size,steps,
                            detector_lambdas=(.25,.5,1.,2.),restarts=2,use_surrogate_logits=False):
    """Select the successful targeted restart with the lowest surrogate score per sample."""
    best=images.detach().clone();best_score=torch.full((len(images),),float("inf"),device=images.device)
    for lam in detector_lambdas:
        for _ in range(restarts):
            candidate=adaptive_pgd(model,extractor,surrogate,images,targets,epsilon,step_size,steps,float(lam),
                                   use_surrogate_logits=use_surrogate_logits)
            with torch.no_grad():
                extractor.reset();pred=model(candidate).argmax(1);score=surrogate(extractor.matrix(),targets)
                better=pred.eq(targets)&(score<best_score);best[better]=candidate[better];best_score[better]=score[better]
    return best


def adaptive_typicality_pgd(model,extractor,surrogate,images,targets,epsilon,step_size,steps,
                            signature_center,signature_scale,energy_center,energy_scale,
                            msp_center,msp_scale,signature_lambda=1.,energy_lambda=1.,
                            msp_lambda=1.,random_start=True):
    """Targeted PGD that directly tries to enter both clean typicality regions.

    Centers/scales are computed on clean calibration data in surrogate-logit and
    energy space.  Smooth-L1 penalties avoid the old one-sided objective's
    incentive to drive the signature score to an unnaturally low value.
    """
    x0=images.detach();x=x0+torch.empty_like(x0).uniform_(-epsilon,epsilon) if random_start else x0.clone()
    x=x.clamp(0,1).detach();signature_center=signature_center.to(x);signature_scale=signature_scale.to(x)
    energy_center=energy_center.to(x);energy_scale=energy_scale.to(x);msp_center=msp_center.to(x);msp_scale=msp_scale.to(x)
    for _ in range(steps):
        x.requires_grad_(True);extractor.reset();logits=model(x);surrogate_logit=surrogate.logits(extractor.matrix(),targets)
        sig_z=(surrogate_logit-signature_center[targets])/signature_scale[targets].clamp_min(1e-4)
        energy=-torch.logsumexp(logits,1);energy_z=(energy-energy_center[targets])/energy_scale[targets].clamp_min(1e-4)
        msp=1-logits.softmax(1).max(1).values;msp_z=(msp-msp_center[targets])/msp_scale[targets].clamp_min(1e-4)
        sig_penalty=F.smooth_l1_loss(sig_z,torch.zeros_like(sig_z),reduction="none")
        energy_penalty=F.smooth_l1_loss(energy_z,torch.zeros_like(energy_z),reduction="none")
        msp_penalty=F.smooth_l1_loss(msp_z,torch.zeros_like(msp_z),reduction="none")
        per=(F.cross_entropy(logits,targets,reduction="none")+signature_lambda*sig_penalty+
             energy_lambda*energy_penalty+msp_lambda*msp_penalty)
        grad=torch.autograd.grad(per.sum(),x)[0]
        if not torch.isfinite(grad).all() or grad.abs().max()==0:raise RuntimeError("Typicality-aware attack gradient invalid")
        x=x.detach()-step_size*grad.sign();x=torch.max(torch.min(x,x0+epsilon),x0-epsilon).clamp(0,1)
    return x.detach()


def adaptive_typicality_pgd_multistart(model,extractor,surrogate,images,targets,epsilon,step_size,steps,
                                       signature_center,signature_scale,energy_center,energy_scale,
                                       msp_center,msp_scale,weight_triples=((.05,.05,.05),(.2,.1,.1),(.1,.2,.2))):
    """Keep the successful targeted candidate closest to both clean centers."""
    best=images.detach().clone();best_objective=torch.full((len(images),),float("inf"),device=images.device)
    for signature_lambda,energy_lambda,msp_lambda in weight_triples:
        candidate=adaptive_typicality_pgd(
            model,extractor,surrogate,images,targets,epsilon,step_size,steps,
            signature_center,signature_scale,energy_center,energy_scale,msp_center,msp_scale,
            signature_lambda,energy_lambda,msp_lambda,True)
        with torch.no_grad():
            extractor.reset();logits=model(candidate);pred=logits.argmax(1);surrogate_logit=surrogate.logits(extractor.matrix(),targets)
            sig_z=((surrogate_logit-signature_center.to(candidate)[targets])/signature_scale.to(candidate)[targets].clamp_min(1e-4)).abs()
            energy=-torch.logsumexp(logits,1)
            energy_z=((energy-energy_center.to(candidate)[targets])/energy_scale.to(candidate)[targets].clamp_min(1e-4)).abs()
            msp=1-logits.softmax(1).max(1).values
            msp_z=((msp-msp_center.to(candidate)[targets])/msp_scale.to(candidate)[targets].clamp_min(1e-4)).abs()
            objective=sig_z+energy_z+msp_z;better=pred.eq(targets)&(objective<best_objective)
            best[better]=candidate[better];best_objective[better]=objective[better]
    return best.detach()


def surrogate_quality(pred, real):
    pred, real = np.asarray(pred), np.asarray(real)
    corr = float(np.corrcoef(pred, real)[0, 1]) if len(pred) > 1 and pred.std()>0 and real.std()>0 else 0.0
    return {"pearson": corr, "rmse": float(np.sqrt(np.mean((pred-real)**2)))}
