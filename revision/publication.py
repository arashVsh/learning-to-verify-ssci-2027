"""Locked, resumable publication protocol for the signature-guided verifier.

This module intentionally uses a result tree separate from the exploratory
revision stages.  A final marker is written only after the corresponding
artifact has been atomically saved and validated.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import pickle
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT",str(max(1,min(4,os.cpu_count() or 1))))
warnings.filterwarnings("ignore",message=r"dtype\(\): align should be passed as Python or NumPy boolean.*")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import SGDClassifier
from scipy.stats import t as student_t
from torch.utils.data import DataLoader,Subset,TensorDataset
from torchvision import datasets,models,transforms

from revision.core import (CrossBankGeometry,GraphActivationExtractor,NormalizedModel,
    adaptive_baseline_pgd,adaptive_verifier_apgd,adaptive_verifier_cw,
    adaptive_verifier_l2_pgd,adaptive_verifier_pgd,atomic_json,atomic_torch_save,
    choose_targets,class_conditional_thresholds,completed,
    mark_complete,metrics_class_conditional,ordinary_targeted_pgd,seed_everything,
    targeted_success,verifier_training_attack)


DATASETS=("cifar10","svhn","stl10")
ARCHITECTURES=("resnet18","densenet121","convnext_tiny")
ATTACKS=("joint_pgd_linf","joint_apgd_linf","joint_cw_l2","autoattack_candidate_refine_linf")
BASELINES=("msp","energy","diagonal_mahalanobis","activation_logistic")
ABLATIONS=("no_signature","no_distillation","no_adaptive_training")
MIN_CLASSIFIER_ACCURACY={"cifar10":.85,"svhn":.90,"stl10":.65}
STATS={
    "cifar10":((.4914,.4822,.4465),(.2470,.2435,.2616),10,32),
    "svhn":((.4377,.4438,.4728),(.1980,.2010,.1970),10,32),
    "stl10":((.4467,.4398,.4066),(.2603,.2566,.2713),10,96),
}


@dataclass(frozen=True)
class PublicationConfig:
    dataset:str="cifar10"
    architecture:str="resnet18"
    seed:int=0
    root:str="Results/publication_v1"
    checkpoint_root:str="checkpoints/publication_v1"
    classifier_epochs:int=50
    verifier_epochs:int=20
    classifier_batch_size:int=128
    verifier_batch_size:int=32
    attack_batch_size:int=32
    inner_steps:int=7
    attack_steps:int=80
    cw_steps:int=160
    epsilon_linf:float=8/255
    step_linf:float=2/255
    epsilon_l2:float=1.0
    step_l2:float=.1
    cw_l2_limit:float=3.0
    guidance_per_class:int=300
    guidance_adv_per_target:int=150
    verifier_train_per_class:int=1000
    calibration_per_class:int=200
    development_per_class:int=100
    final_clean_per_class:int=200
    final_attack_per_target:int=100
    signature_candidates:int=8192
    signature_top_k:int=256
    n_last_layers:int=8
    num_workers:int=2
    variant:str="full"
    smoke:bool=False
    study_profile:str="minimum"

    @property
    def run_dir(self):return Path(self.root)/self.dataset/self.architecture/f"seed_{self.seed}"
    @property
    def classifier_checkpoint(self):return Path(self.checkpoint_root)/self.dataset/self.architecture/f"seed_{self.seed}.pt"
    @property
    def n_classes(self):return STATS[self.dataset][2]


def smoke_config(cfg):
    return replace(cfg,root="Results/publication_smoke",checkpoint_root="checkpoints/publication_smoke",
                   classifier_epochs=3,verifier_epochs=1,classifier_batch_size=8,verifier_batch_size=4,
                   attack_batch_size=2,inner_steps=1,attack_steps=2,cw_steps=2,
                   epsilon_linf=.5,step_linf=.25,epsilon_l2=10.,step_l2=2.,cw_l2_limit=20.,
                   guidance_per_class=2,guidance_adv_per_target=1,verifier_train_per_class=2,
                   calibration_per_class=2,development_per_class=2,final_clean_per_class=2,
                   final_attack_per_target=1,signature_candidates=32,signature_top_k=4,num_workers=0,smoke=True)


def resource_adjusted_config(cfg):
    """Conservative batches for dual-model gradients on an 8 GB laptop GPU."""
    if cfg.smoke:return cfg
    if cfg.dataset=="stl10":return replace(cfg,classifier_batch_size=min(cfg.classifier_batch_size,32),
        verifier_batch_size=min(cfg.verifier_batch_size,16),attack_batch_size=min(cfg.attack_batch_size,8))
    if cfg.architecture in ("densenet121","convnext_tiny"):
        return replace(cfg,classifier_batch_size=min(cfg.classifier_batch_size,64),
                       verifier_batch_size=min(cfg.verifier_batch_size,32),attack_batch_size=min(cfg.attack_batch_size,16))
    return cfg


def safe_cuda_cleanup():
    """Best-effort cleanup that never masks the original CUDA exception."""
    gc.collect()
    if not torch.cuda.is_available():return
    try:torch.cuda.empty_cache()
    except Exception as exc:print(f"[warning] CUDA cache cleanup skipped after accelerator failure: {exc}",flush=True)


def is_cuda_oom(exc):
    message=f"{type(exc).__name__}: {exc}".lower()
    return "cuda" in message and ("out of memory" in message or "memoryallocation" in message)


def resolve_device(requested="auto"):
    """Resolve and validate a CPU/CUDA device without silent fallback."""
    requested=requested.strip().lower()
    if requested=="auto":requested="cuda:0" if torch.cuda.is_available() else "cpu"
    device=torch.device(requested)
    if device.type=="cuda":
        if not torch.cuda.is_available():raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
        index=torch.cuda.current_device() if device.index is None else device.index
        count=torch.cuda.device_count()
        if index<0 or index>=count:
            raise RuntimeError(f"Requested cuda:{index}, but PyTorch exposes {count} CUDA device(s): valid indices are 0..{count-1}")
        torch.cuda.set_device(index);device=torch.device(f"cuda:{index}")
        print(f"[device] Using {device}: {torch.cuda.get_device_name(index)}",flush=True)
    else:print("[device] Using CPU",flush=True)
    return device


def dataset_object(name,train,augment=False):
    _,_,_,size=STATS[name];root=Path("data")
    aug=[]
    if augment:
        aug=[transforms.RandomCrop(size,padding=4 if size==32 else 8)]
        if name!="svhn":aug.append(transforms.RandomHorizontalFlip())
    transform=transforms.Compose(aug+[transforms.ToTensor()])
    if name=="cifar10":return datasets.CIFAR10(root,train=train,transform=transform,download=True)
    if name=="svhn":return datasets.SVHN(root,split="train" if train else "test",transform=transform,download=True)
    if name=="stl10":return datasets.STL10(root,split="train" if train else "test",transform=transform,download=True)
    raise ValueError(name)


def labels_array(ds):
    if hasattr(ds,"targets"):return np.asarray(ds.targets,dtype=int)
    if hasattr(ds,"labels"):return np.asarray(ds.labels,dtype=int)
    return np.asarray([int(ds[i][1]) for i in range(len(ds))],dtype=int)


def detector_splits(ds,seed,n_classes=10):
    labels=labels_array(ds);rng=np.random.default_rng(seed+19001);parts={k:[] for k in ("guidance","verifier","development","calibration")}
    for cls in range(n_classes):
        idx=np.where(labels==cls)[0];rng.shuffle(idx);a=int(.2*len(idx));b=int(.7*len(idx));c=int(.85*len(idx))
        for key,fold in (("guidance",idx[:a]),("verifier",idx[a:b]),("development",idx[b:c]),("calibration",idx[c:])):
            parts[key].extend(fold.tolist())
    return {k:sorted(v) for k,v in parts.items()}


def effective_per_class(ds,indices,requested,n_classes=10):
    labels=labels_array(ds)[np.asarray(indices,dtype=int)];available=min(int((labels==c).sum()) for c in range(n_classes))
    return min(int(requested),available)


def balanced_tensors(ds,indices,per_class,n_classes=10):
    buckets={c:[] for c in range(n_classes)}
    for i in indices:
        x,y=ds[int(i)];y=int(y)
        if len(buckets[y])<per_class:buckets[y].append(x)
        if min(map(len,buckets.values()))>=per_class:break
    counts={c:len(v) for c,v in buckets.items()}
    if min(counts.values())<per_class:raise RuntimeError(f"Insufficient balanced samples: {counts}; requested={per_class}")
    x=torch.cat([torch.stack(buckets[c]) for c in range(n_classes)])
    y=torch.cat([torch.full((per_class,),c,dtype=torch.long) for c in range(n_classes)])
    return x,y


def build_architecture(name,n_classes):
    if name=="resnet18":
        net=models.resnet18(weights=None);net.conv1=torch.nn.Conv2d(3,64,3,1,1,bias=False);net.maxpool=torch.nn.Identity()
        net.fc=torch.nn.Linear(net.fc.in_features,n_classes)
    elif name=="densenet121":
        net=models.densenet121(weights=None);net.features.conv0=torch.nn.Conv2d(3,64,3,1,1,bias=False);net.features.pool0=torch.nn.Identity()
        net.classifier=torch.nn.Linear(net.classifier.in_features,n_classes)
    elif name=="convnext_tiny":
        net=models.convnext_tiny(weights=None);net.classifier[-1]=torch.nn.Linear(net.classifier[-1].in_features,n_classes)
    else:raise ValueError(name)
    return net


def new_model(cfg,device):
    mean,std,_,_=STATS[cfg.dataset]
    return NormalizedModel(build_architecture(cfg.architecture,cfg.n_classes),mean,std).to(device)


def load_classifier(cfg,device):
    path=cfg.classifier_checkpoint
    if not completed(path,False):raise RuntimeError(f"Classifier incomplete: {path}. Use --train-missing-classifiers.")
    saved=torch.load(path,map_location="cpu",weights_only=False)
    if not cfg.smoke and float(saved.get("official_test_accuracy",-1))<MIN_CLASSIFIER_ACCURACY[cfg.dataset]:
        raise RuntimeError(f"Classifier at {path} is below the locked {MIN_CLASSIFIER_ACCURACY[cfg.dataset]:.0%} accuracy gate")
    model=new_model(cfg,device);model.load_state_dict(saved["state_dict"]);model.eval();return model,saved


@torch.no_grad()
def batched_logits(model,x,device,batch=128):
    return torch.cat([model(x[i:i+batch].to(device)).cpu() for i in range(0,len(x),batch)])


@torch.no_grad()
def accuracy(model,ds,device,batch=256,num_workers=2):
    loader=DataLoader(ds,batch_size=batch,num_workers=num_workers,pin_memory=device.type=="cuda");correct=total=0
    for x,y in loader:
        pred=model(x.to(device,non_blocking=True)).argmax(1).cpu();correct+=pred.eq(y).sum().item();total+=len(y)
    return correct/max(1,total)


def train_classifier(cfg,device,force=False):
    path=cfg.classifier_checkpoint
    if completed(path,force):return torch.load(path,map_location="cpu",weights_only=False)
    seed_everything(cfg.seed);model=new_model(cfg,device);train_ds=dataset_object(cfg.dataset,True,augment=True)
    test_ds=dataset_object(cfg.dataset,False,augment=False)
    splits=detector_splits(train_ds,cfg.seed,cfg.n_classes)
    classifier_indices=sorted(splits["guidance"]+splits["verifier"])
    classifier_data=Subset(train_ds,classifier_indices)
    if cfg.smoke:
        labels=labels_array(train_ds);idx=np.concatenate([np.where(labels==c)[0][:4] for c in range(cfg.n_classes)]).tolist();classifier_data=Subset(train_ds,idx)
    loader=DataLoader(classifier_data,batch_size=cfg.classifier_batch_size,shuffle=True,num_workers=cfg.num_workers,
                      pin_memory=device.type=="cuda",persistent_workers=cfg.num_workers>0)
    if cfg.architecture=="convnext_tiny":opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
    else:opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4,nesterov=True)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.classifier_epochs);start=0
    if path.exists() and not force:
        saved=torch.load(path,map_location="cpu",weights_only=False);model.load_state_dict(saved["state_dict"])
        if "optimizer" in saved:opt.load_state_dict(saved["optimizer"])
        if "scheduler" in saved:scheduler.load_state_dict(saved["scheduler"])
        start=int(saved.get("epoch",0));print(f"[classifier] Resume {cfg.dataset}/{cfg.architecture}/seed{cfg.seed} epoch {start}",flush=True)
    amp=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    for epoch in range(start,cfg.classifier_epochs):
        model.train();correct=total=0;losses=[];started=time.time()
        for i,(x,y) in enumerate(loader):
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=device.type=="cuda"):
                logits=model(x);loss=F.cross_entropy(logits,y,label_smoothing=.05)
            amp.scale(loss).backward();amp.step(opt);amp.update();losses.append(float(loss.detach()))
            correct+=logits.argmax(1).eq(y).sum().item();total+=len(y)
            if i%100==0:print(f"[classifier {epoch+1}/{cfg.classifier_epochs}] batch={i}/{len(loader)} loss={losses[-1]:.4f}",flush=True)
        scheduler.step();payload={"state_dict":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),
            "optimizer_name":type(opt).__name__,"epoch":epoch+1,"configuration":asdict(cfg),
            "train_accuracy":correct/total,"mean_loss":float(np.mean(losses))}
        atomic_torch_save(payload,path);print(f"[classifier] epoch={epoch+1} acc={correct/total:.4f} elapsed={(time.time()-started)/60:.1f}m",flush=True)
    model.eval();payload=torch.load(path,map_location="cpu",weights_only=False)
    payload["official_test_accuracy"]=accuracy(model,test_ds,device,num_workers=cfg.num_workers)
    payload["training_partition"]="70% stratified official-train guidance+verifier partitions; development and calibration excluded"
    payload["training_samples"]=len(classifier_data)
    payload["minimum_accuracy_gate"]=None if cfg.smoke else MIN_CLASSIFIER_ACCURACY[cfg.dataset]
    atomic_torch_save(payload,path)
    if not cfg.smoke and payload["official_test_accuracy"]<MIN_CLASSIFIER_ACCURACY[cfg.dataset]:
        raise RuntimeError(f"Classifier accuracy {payload['official_test_accuracy']:.4f} is below the locked "
                           f"{MIN_CLASSIFIER_ACCURACY[cfg.dataset]:.4f} gate; checkpoint retained for resumable diagnosis")
    mark_complete(path)
    return payload


@torch.no_grad()
def activation_matrix(model,x,cfg,device,batch=64):
    extractor=GraphActivationExtractor(model,cfg.n_last_layers);parts=[]
    try:
        for i in range(0,len(x),batch):
            z=x[i:i+batch].to(device);extractor.reset();model(z);parts.append(extractor.matrix().cpu())
    finally:extractor.close()
    return torch.cat(parts)


def cached_pool(path,factory,force=False):
    if completed(path,force):
        print(f"[cache] {path}",flush=True);return torch.load(path,map_location="cpu",weights_only=False)
    value=factory();atomic_torch_save(value,path);mark_complete(path);return value


def _balanced_protocol_targets(labels,n_classes,seed,start_index,assigned_counts):
    """Assign non-true targets as evenly as possible, independent of attack success."""
    targets=[]
    for offset,true_class in enumerate(labels.detach().cpu().tolist()):
        allowed=[c for c in range(n_classes) if c!=int(true_class)]
        least=min(int(assigned_counts[c]) for c in allowed)
        candidates=[c for c in allowed if int(assigned_counts[c])==least]
        target=candidates[(int(seed)+int(start_index)+offset)%len(candidates)]
        assigned_counts[target]+=1;targets.append(target)
    return torch.tensor(targets,device=labels.device,dtype=labels.dtype)


def _smoke_pool(attack,model,loader,per_target,n_classes,seed,device,label):
    """Build a balanced structural-test pool without asserting attack success.

    A smoke classifier is deliberately under-trained, so requiring ten targeted
    successes would make the software test depend on model quality.  Scientific
    runs never call this function and retain the strict successful-only gate.
    """
    buckets={c:{"x":[],"original":[],"y":[],"t":[]} for c in range(n_classes)}
    attempted=successful=0;assigned_counts=np.zeros(n_classes,dtype=np.int64)
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        targets=_balanced_protocol_targets(y,n_classes,seed,attempted,assigned_counts);adversarial=attack(x,targets)
        ok=targeted_success(model,adversarial,targets);attempted+=len(x);successful+=int(ok.sum())
        for cls in range(n_classes):
            room=per_target-sum(len(z) for z in buckets[cls]["x"])
            take=torch.where(targets.eq(cls))[0][:max(0,room)]
            if len(take):
                buckets[cls]["x"].append(adversarial[take].detach().cpu())
                buckets[cls]["original"].append(x[take].detach().cpu())
                buckets[cls]["y"].append(y[take].cpu());buckets[cls]["t"].append(targets[take].cpu())
        if min(sum(len(z) for z in buckets[c]["x"]) for c in range(n_classes))>=per_target:break
    counts=[sum(len(z) for z in buckets[c]["x"]) for c in range(n_classes)]
    if min(counts)<per_target:raise RuntimeError(f"{label}: structural pool exhausted; per-target counts={counts}")
    joined=lambda key:torch.cat([torch.cat(buckets[c][key])[:per_target] for c in range(n_classes)])
    return {"adversarial":joined("x"),"original":joined("original"),"true_labels":joined("y"),
            "targets":joined("t"),"attempted":attempted,"successful":successful,"retained":per_target*n_classes,
            "targeted_asr":successful/max(1,attempted),"retained_per_target":{str(c):per_target for c in range(n_classes)},
            "structural_test_only":True}


def collect_protocol_pool(cfg,attack,model,loader,per_target,seed,device,label,allow_balanced_shortfall=False):
    """Collect a target-balanced pool of successful attacks.

    Evaluation pools keep an exact locked quota. Guidance pools may opt into a
    balanced shortfall because their finite training split can contain fewer
    correctly classified inputs than the requested total. In that case all
    target buckets are trimmed to the same observed count, and a 50% floor
    prevents a weak classifier or weak attack from being accepted silently.
    """
    if cfg.smoke:return _smoke_pool(attack,model,loader,per_target,cfg.n_classes,seed,device,label)
    buckets={c:{"x":[],"original":[],"y":[],"t":[]} for c in range(cfg.n_classes)}
    seen=eligible=successful=0;started=time.time();assigned_counts=np.zeros(cfg.n_classes,dtype=np.int64)
    for batch_i,(x,y) in enumerate(loader):
        x,y=x.to(device),y.to(device)
        with torch.no_grad():clean_ok=model(x).argmax(1).eq(y)
        x,y=x[clean_ok],y[clean_ok];seen+=len(clean_ok);eligible+=len(x)
        if not len(x):continue
        targets=_balanced_protocol_targets(y,cfg.n_classes,seed,eligible-len(x),assigned_counts)
        adversarial=attack(x,targets);ok=targeted_success(model,adversarial,targets);successful+=int(ok.sum())
        for cls in range(cfg.n_classes):
            room=per_target-sum(len(z) for z in buckets[cls]["x"])
            take=torch.where(ok&targets.eq(cls))[0][:max(0,room)]
            if len(take):
                buckets[cls]["x"].append(adversarial[take].detach().cpu());buckets[cls]["original"].append(x[take].detach().cpu())
                buckets[cls]["y"].append(y[take].cpu());buckets[cls]["t"].append(targets[take].cpu())
        counts=[sum(len(z) for z in buckets[c]["x"]) for c in range(cfg.n_classes)]
        print(f"[{time.strftime('%H:%M:%S')}][{label}] seen={seen} eligible={eligible} successful={successful} "
              f"per_target={min(counts)}/{per_target} elapsed={(time.time()-started)/60:.1f}m",flush=True)
        if min(counts)>=per_target:break
    counts=[sum(len(z) for z in buckets[c]["x"]) for c in range(cfg.n_classes)]
    retained_per_target=per_target
    if min(counts)<per_target:
        if not allow_balanced_shortfall:
            raise RuntimeError(f"{label}: exhausted correctly classified candidates; per-target counts={counts}")
        retained_per_target=min(counts);minimum=max(1,int(np.ceil(.5*per_target)))
        if retained_per_target<minimum:
            raise RuntimeError(f"{label}: balanced guidance shortfall is too large; per-target counts={counts}, "
                               f"minimum acceptable per target={minimum}")
        print(f"[{label}] using largest available balanced guidance pool: "
              f"{retained_per_target}/{per_target} successful examples per target",flush=True)
    joined=lambda key:torch.cat([torch.cat(buckets[c][key])[:retained_per_target] for c in range(cfg.n_classes)])
    return {"adversarial":joined("x"),"original":joined("original"),"true_labels":joined("y"),"targets":joined("t"),
            "seen":seen,"eligible_correctly_classified":eligible,"successful":successful,
            "retained":retained_per_target*cfg.n_classes,"requested_per_target":per_target,
            "balanced_shortfall":retained_per_target<per_target,
            "targeted_asr_on_eligible":successful/max(1,eligible),
            "retained_per_target":{str(c):retained_per_target for c in range(cfg.n_classes)}}


def _robust_centers(values,predicted,n_classes,min_scale=.05):
    values=np.asarray(values,dtype=np.float32);predicted=np.asarray(predicted);centers=[];scales=[]
    gc=float(np.median(values));gs=float(1.4826*np.median(np.abs(values-gc)))
    for cls in range(n_classes):
        z=values[predicted==cls];c=float(np.median(z)) if len(z)>=10 else gc
        s=float(1.4826*np.median(np.abs(z-c))) if len(z)>=10 else gs;centers.append(c);scales.append(max(min_scale,s))
    return torch.tensor(centers),torch.tensor(scales)


def geometry_from_artifact(a,device):
    return CrossBankGeometry(a["indices"],a["centers"],a["scales"],a["distance_center"],a["distance_scale"],
                             a["margin_center"],a["margin_scale"],a.get("temperature",.25)).to(device).eval()


def train_signature_guidance(cfg,model,train_ds,splits,device,force=False):
    root=cfg.run_dir/"guidance";path=root/"signature_guidance.pt"
    if completed(path,force):
        a=torch.load(path,map_location="cpu",weights_only=False);return a,geometry_from_artifact(a,device)
    guide_n=effective_per_class(train_ds,splits["guidance"],cfg.guidance_per_class,cfg.n_classes)
    clean_x,clean_y=balanced_tensors(train_ds,splits["guidance"],guide_n,cfg.n_classes)
    loader=DataLoader(Subset(train_ds,splits["guidance"]),batch_size=cfg.attack_batch_size,shuffle=False)
    attack=lambda x,t:ordinary_targeted_pgd(model,x,t,cfg.epsilon_linf,cfg.step_linf,max(10,cfg.inner_steps))
    guide_adv_n=min(cfg.guidance_adv_per_target,guide_n)
    pool=cached_pool(root/"guidance_targeted_pgd.pt",lambda:collect_protocol_pool(
        cfg,attack,model,loader,guide_adv_n,cfg.seed+2001,device,"publication-guidance-PGD",
        allow_balanced_shortfall=True),force)
    clean_acts=activation_matrix(model,clean_x,cfg,device);adv_acts=activation_matrix(model,pool["adversarial"],cfg,device)
    variance=torch.cat([clean_acts,adv_acts]).var(0);candidate=variance.topk(min(cfg.signature_candidates,len(variance))).indices
    indices=[];centers=[];scales=[]
    for cls in range(cfg.n_classes):
        c=clean_acts[clean_y.eq(cls)][:,candidate];a=adv_acts[pool["targets"].eq(cls)][:,candidate]
        X=torch.cat([c,a]).numpy();labels=np.r_[np.zeros(len(c),dtype=int),np.ones(len(a),dtype=int)]
        mi=mutual_info_classif(X,labels,random_state=cfg.seed+cls,n_neighbors=min(3,len(X)-1))
        local=np.argsort(mi)[-min(cfg.signature_top_k,len(mi)):][::-1].copy();idx=candidate[torch.tensor(local)]
        z=clean_acts[clean_y.eq(cls)][:,idx];center=z.median(0).values;scale=(z-center).abs().median(0).values*1.4826
        floor=max(1e-3,float(torch.quantile(scale[scale>0],.1))*.1) if (scale>0).any() else 1e-3
        indices.append(idx);centers.append(center);scales.append(scale.clamp_min(floor))
    indices=torch.stack(indices);centers=torch.stack(centers);scales=torch.stack(scales)
    provisional=CrossBankGeometry(indices,centers,scales,torch.zeros(cfg.n_classes),torch.ones(cfg.n_classes),
                                  torch.zeros(cfg.n_classes),torch.ones(cfg.n_classes),.25).to(device)
    pred=batched_logits(model,clean_x,device).argmax(1).numpy();chosen=[];margins=[]
    with torch.no_grad():
        for i in range(0,len(clean_acts),128):
            p=torch.tensor(pred[i:i+128],device=device);_,c,m=provisional.components(clean_acts[i:i+128].to(device),p)
            chosen.append(c.cpu());margins.append(m.cpu())
    dc,ds=_robust_centers(torch.cat(chosen).numpy(),pred,cfg.n_classes);mc,ms=_robust_centers(torch.cat(margins).numpy(),pred,cfg.n_classes)
    artifact={"indices":indices,"centers":centers,"scales":scales,"distance_center":dc,"distance_scale":ds,
              "margin_center":mc,"margin_scale":ms,"temperature":.25,"candidate_count":len(candidate),
              "top_k":indices.shape[1],"configuration":asdict(cfg),"fit_partition":"training guidance split",
              "adversarial_guidance":"successful targeted Linf PGD only"}
    atomic_torch_save(artifact,path);mark_complete(path);return artifact,geometry_from_artifact(artifact,device)


def augment_pixels(x,dataset):
    pad=4 if STATS[dataset][3]==32 else 8;x=F.pad(x,(pad,pad,pad,pad),mode="reflect");size=STATS[dataset][3];out=[]
    for z in x:
        top=int(torch.randint(0,2*pad+1,()).item());left=int(torch.randint(0,2*pad+1,()).item());z=z[:,top:top+size,left:left+size]
        if dataset!="svhn" and bool(torch.rand(())<.5):z=torch.flip(z,(-1,))
        out.append(z)
    return torch.stack(out)


def _load_verifier(model,path,device):
    saved=torch.load(path,map_location="cpu",weights_only=False);verifier=copy.deepcopy(model).to(device)
    verifier.load_state_dict(saved.get("best_state_dict",saved["state_dict"]));verifier.eval();return verifier,saved


@torch.no_grad()
def verifier_values(model,verifier,x,device,batch=128):
    values=[];predicted=[];verifier_pred=[]
    for i in range(0,len(x),batch):
        z=x[i:i+batch].to(device);base=model(z);verify=verifier(z);pred=base.argmax(1);prob=verify.softmax(1);row=torch.arange(len(z),device=device)
        values.append((1-prob[row,pred]).cpu());predicted.append(pred.cpu());verifier_pred.append(verify.argmax(1).cpu())
    return torch.cat(values).numpy(),torch.cat(predicted).numpy(),torch.cat(verifier_pred).numpy()


@torch.no_grad()
def verifier_latency_ms(model,verifier,x,device,batch=64):
    x=x[:min(len(x),256)]
    if not len(x):return None
    warm=x[:min(batch,len(x))].to(device);model(warm);verifier(warm)
    if device.type=="cuda":torch.cuda.synchronize()
    started=time.perf_counter()
    for i in range(0,len(x),batch):
        z=x[i:i+batch].to(device);base=model(z);verify=verifier(z);_=(1-verify.softmax(1)[torch.arange(len(z),device=device),base.argmax(1)]).cpu()
    if device.type=="cuda":torch.cuda.synchronize()
    return 1000*(time.perf_counter()-started)/len(x)


def _development_score(cfg,model,verifier,x,y,device):
    clean_pred=batched_logits(verifier,x,device).argmax(1);clean_acc=float(clean_pred.eq(y).float().mean())
    n=min(64,len(x));source=x[:n].to(device);labels=y[:n].to(device);targets=choose_targets(labels,cfg.n_classes,cfg.seed+31001)
    attack=adaptive_verifier_pgd(model,verifier,source,targets,cfg.epsilon_linf,cfg.step_linf,min(10,cfg.attack_steps),(.2,1.,5.))
    with torch.no_grad():robust=float(verifier(attack).argmax(1).eq(labels).float().mean())
    return clean_acc,robust,.5*(clean_acc+robust)


def train_verifier(cfg,model,geometry,train_ds,splits,device,variant="full",force=False):
    root=cfg.run_dir/("verifier" if variant=="full" else f"ablations/{variant}");path=root/"verifier.pt"
    if completed(path,force):return _load_verifier(model,path,device)
    train_n=effective_per_class(train_ds,splits["verifier"],cfg.verifier_train_per_class,cfg.n_classes)
    dev_n=effective_per_class(train_ds,splits["development"],cfg.development_per_class,cfg.n_classes)
    train_x,train_y=balanced_tensors(train_ds,splits["verifier"],train_n,cfg.n_classes)
    dev_x,dev_y=balanced_tensors(train_ds,splits["development"],dev_n,cfg.n_classes)
    verifier=copy.deepcopy(model).to(device);loader=DataLoader(TensorDataset(train_x,train_y),batch_size=cfg.verifier_batch_size,shuffle=True)
    if cfg.architecture=="convnext_tiny":opt=torch.optim.AdamW(verifier.parameters(),lr=1e-4,weight_decay=.01)
    else:opt=torch.optim.SGD(verifier.parameters(),lr=.01,momentum=.9,weight_decay=5e-4,nesterov=True)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.verifier_epochs);start=0;best_score=-1.;best_state=None;history=[];resume_events=[]
    if path.exists() and not force:
        saved=torch.load(path,map_location="cpu",weights_only=False);verifier.load_state_dict(saved["state_dict"]);start=int(saved.get("epoch",0))
        best_score=float(saved.get("best_score",-1));best_state=saved.get("best_state_dict");history=saved.get("history",[])
        resume_events=saved.get("resume_events",[]);saved_batch=int(saved.get("configuration",{}).get("verifier_batch_size",cfg.verifier_batch_size))
        if saved_batch!=cfg.verifier_batch_size:
            event={"resumed_after_epoch":start,"old_verifier_batch_size":saved_batch,"new_verifier_batch_size":cfg.verifier_batch_size,
                   "reason":"resource adjustment or automatic CUDA OOM retry"}
            if not resume_events or resume_events[-1]!=event:resume_events.append(event)
        if "optimizer" in saved:opt.load_state_dict(saved["optimizer"])
        if "scheduler" in saved:scheduler.load_state_dict(saved["scheduler"])
        print(f"[verifier {variant}] Resume epoch {start}/{cfg.verifier_epochs}",flush=True)
    geometry_weight=0. if variant in ("no_signature","no_adaptive_training") else .05
    extractor=GraphActivationExtractor(model,cfg.n_last_layers) if geometry_weight>0 else None;model.eval()
    try:
        for epoch in range(start,cfg.verifier_epochs):
            verifier.train();losses=[];clean_correct=adv_correct=total=0;started=time.time()
            for batch,(x,y) in enumerate(loader):
                x,y=augment_pixels(x,cfg.dataset).to(device),y.to(device);targets=choose_targets(y,cfg.n_classes,cfg.seed+epoch*100000+batch*len(y))
                if variant=="no_adaptive_training":adv=x.detach()
                else:
                    verifier.eval();adv=verifier_training_attack(model,verifier,extractor,geometry,x,y,targets,
                        cfg.epsilon_linf,cfg.step_linf,cfg.inner_steps,1.,geometry_weight)
                    if extractor is not None:extractor.reset()
                verifier.train();clean_logits=verifier(x)
                adv_logits=clean_logits if variant=="no_adaptive_training" else verifier(adv)
                loss_clean=F.cross_entropy(clean_logits,y,label_smoothing=.05)
                loss_adv=torch.zeros((),device=device) if variant=="no_adaptive_training" else F.cross_entropy(adv_logits,y,label_smoothing=.05)
                if variant=="no_distillation":distill=torch.zeros((),device=device)
                else:
                    with torch.no_grad():teacher=model(x)
                    if extractor is not None:extractor.reset()
                    distill=F.kl_div(F.log_softmax(clean_logits,1),F.softmax(teacher,1),reduction="batchmean")
                loss=loss_clean+1.5*loss_adv+.5*distill;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(verifier.parameters(),5.);opt.step()
                clean_correct+=clean_logits.argmax(1).eq(y).sum().item();adv_correct+=adv_logits.argmax(1).eq(y).sum().item();total+=len(y);losses.append(float(loss.detach()))
                if batch%25==0:print(f"[verifier {variant} {epoch+1}/{cfg.verifier_epochs}] batch={batch}/{len(loader)} loss={losses[-1]:.4f}",flush=True)
            scheduler.step();verifier.eval();dev_clean,dev_robust,score=_development_score(cfg,model,verifier,dev_x,dev_y,device)
            row={"epoch":epoch+1,"loss":float(np.mean(losses)),"train_clean_accuracy":clean_correct/total,
                 "train_adversarial_accuracy":adv_correct/total,"development_clean_accuracy":dev_clean,
                 "development_robust_accuracy":dev_robust,"selection_score":score,"elapsed_minutes":(time.time()-started)/60}
            history.append(row);print(f"[verifier {variant}] {row}",flush=True)
            if score>best_score:best_score=score;best_state={k:v.detach().cpu().clone() for k,v in verifier.state_dict().items()}
            payload={"state_dict":verifier.state_dict(),"best_state_dict":best_state,"best_score":best_score,"optimizer":opt.state_dict(),
                     "scheduler":scheduler.state_dict(),"optimizer_name":type(opt).__name__,"epoch":epoch+1,"history":history,
                     "resume_events":resume_events,"variant":variant,"configuration":asdict(cfg),
                     "checkpoint_selection":"clean + directly attacked development split; official test untouched"}
            atomic_torch_save(payload,path)
        mark_complete(path)
        verifier.load_state_dict(best_state);verifier.eval()
        return verifier,payload
    finally:
        if extractor is not None:extractor.close()


def perturbation_stats(pool,norm,limit):
    delta=(pool["adversarial"]-pool["original"]).flatten(1)
    values=delta.abs().max(1).values if norm=="linf" else delta.norm(2,1)
    return {"norm":norm,"limit":float(limit),"mean":float(values.mean()),"max":float(values.max()),
            "violations":int((values>limit+1e-5).sum())}


def verifier_attack(name,cfg,model,verifier,weights=None):
    if name=="joint_pgd_linf":
        w=(.05,.2,1.,5.,20.) if weights is None else weights
        return lambda x,t:adaptive_verifier_pgd(model,verifier,x,t,cfg.epsilon_linf,cfg.step_linf,cfg.attack_steps,w),"linf",cfg.epsilon_linf
    if name=="joint_apgd_linf":
        w=(.05,.2,1.,5.) if weights is None else weights
        return lambda x,t:adaptive_verifier_apgd(model,verifier,x,t,cfg.epsilon_linf,cfg.attack_steps,w),"linf",cfg.epsilon_linf
    if name=="joint_pgd_l2":
        w=(.05,.2,1.,5.) if weights is None else weights
        return lambda x,t:adaptive_verifier_l2_pgd(model,verifier,x,t,cfg.epsilon_l2,cfg.step_l2,cfg.attack_steps,w),"l2",cfg.epsilon_l2
    if name=="joint_cw_l2":
        w=(.25,1.,4.) if weights is None else weights
        return lambda x,t:adaptive_verifier_cw(model,verifier,x,t,cfg.cw_steps,cfg.cw_l2_limit,1e-2,w),"l2",cfg.cw_l2_limit
    if name=="autoattack_candidate_refine_linf":
        w=(.05,.2,1.,5.,20.) if weights is None else weights
        def attack(x,t):
            if cfg.smoke:candidate=x.detach()
            else:
                try:import torchattacks
                except ImportError as exc:
                    raise RuntimeError("AutoAttack candidate/refinement requires torchattacks; install requirements.txt") from exc
                clean_labels=model(x).detach().argmax(1)
                generator=torchattacks.AutoAttack(model,norm="Linf",eps=cfg.epsilon_linf,version="standard",
                                                   n_classes=cfg.n_classes,seed=cfg.seed,verbose=False)
                candidate=generator(x,clean_labels).detach()
            return adaptive_verifier_pgd(model,verifier,x,t,cfg.epsilon_linf,cfg.step_linf,cfg.attack_steps,w,initial=candidate)
        return attack,"linf",cfg.epsilon_linf
    raise ValueError(name)


def _attack_validity(name,cfg,model,verifier,pool,device,n=64):
    n=min(n,len(pool["adversarial"]));x=pool["original"][:n].to(device);targets=pool["targets"][:n].to(device)
    restart_count={"joint_pgd_linf":5,"joint_apgd_linf":4,"joint_pgd_l2":4,"joint_cw_l2":3,"autoattack_candidate_refine_linf":5}[name]
    comparator,comparator_norm,comparator_limit=verifier_attack(name,cfg,model,verifier,(0.,)*restart_count)
    ordinary=comparator(x,targets).cpu()
    ok=batched_logits(model,ordinary,device).argmax(1).eq(targets.cpu())
    if not ok.any():return {"passed":False,"failures":[f"classifier-only {name} produced no matched targeted successes"],
                            "comparator_attack":name,"comparator_norm":comparator_norm,"comparator_limit":float(comparator_limit)}
    ordinary_score=verifier_values(model,verifier,ordinary[ok],device)[0]
    adaptive_score=verifier_values(model,verifier,pool["adversarial"][:n][ok],device)[0]
    improvement=float(ordinary_score.mean()-adaptive_score.mean());standardized=improvement/max(1e-4,float(np.std(ordinary_score)))
    fraction=float((adaptive_score<ordinary_score).mean());failures=[]
    if standardized<.1:failures.append("detector objective did not improve by 0.1 ordinary-score standard deviations")
    if fraction<.55:failures.append("detector objective improved on fewer than 55% of matched samples")
    return {"passed":not failures,"failures":failures,"matched_samples":int(ok.sum()),"comparator_attack":name,
            "comparator_norm":comparator_norm,"comparator_limit":float(comparator_limit),
            "mean_objective_improvement":improvement,"standardized_objective_improvement":standardized,"fraction_improved":fraction}


def calibration_and_test(cfg,train_ds,test_ds,splits):
    cal_n=effective_per_class(train_ds,splits["calibration"],cfg.calibration_per_class,cfg.n_classes)
    cal_x,cal_y=balanced_tensors(train_ds,splits["calibration"],cal_n,cfg.n_classes)
    test_idx=np.arange(len(test_ds));test_n=effective_per_class(test_ds,test_idx,cfg.final_clean_per_class,cfg.n_classes)
    test_x,test_y=balanced_tensors(test_ds,test_idx,test_n,cfg.n_classes);return cal_x,cal_y,test_x,test_y


def run_verifier_attacks(cfg,model,verifier,train_ds,test_ds,splits,device,force=False,attack_names=ATTACKS,variant="full"):
    root=cfg.run_dir/("attacks" if variant=="full" else f"ablations/{variant}/attacks");root.mkdir(parents=True,exist_ok=True)
    cal_x,_,test_x,test_y=calibration_and_test(cfg,train_ds,test_ds,splits)
    cal_score,cal_pred,_=verifier_values(model,verifier,cal_x,device);clean_score,clean_pred,verifier_pred=verifier_values(model,verifier,test_x,device)
    thresholds=class_conditional_thresholds(cal_score,cal_pred,.05,cfg.n_classes);loader=DataLoader(test_ds,batch_size=cfg.attack_batch_size,shuffle=False)
    attack_n=min(cfg.final_attack_per_target,effective_per_class(test_ds,np.arange(len(test_ds)),cfg.final_attack_per_target,cfg.n_classes))
    rows={}
    for name in attack_names:
        result_path=root/f"{name}.json"
        if completed(result_path,force):rows[name]=json.loads(result_path.read_text());continue
        started=time.time();attack,norm,limit=verifier_attack(name,cfg,model,verifier);pool=cached_pool(root/f"{name}.pt",lambda a=attack,n=name:collect_protocol_pool(
            cfg,a,model,loader,attack_n,cfg.seed+4101,device,n),force)
        adv_score,adv_pred,_=verifier_values(model,verifier,pool["adversarial"],device)
        row=metrics_class_conditional(clean_score,clean_pred,adv_score,adv_pred,thresholds);validity=_attack_validity(name,cfg,model,verifier,pool,device)
        if cfg.smoke:
            validity={**validity,"passed":False,"failures":[*validity.get("failures",[]),"smoke artifacts are structural tests only"]}
        perturb=perturbation_stats(pool,norm,limit)
        if perturb["violations"]:validity["passed"]=False;validity["failures"].append("norm violation")
        row.update({"attack":name,"detector_aware":True,"seen_during_verifier_training":name=="joint_pgd_linf",
                    "attack_validity":validity,"perturbation":perturb,
                    "attack_yield":{k:v for k,v in pool.items() if not torch.is_tensor(v)},
                    "estimated_joint_target_success_and_bypass_rate":pool.get("targeted_asr_on_eligible",pool.get("targeted_asr",0))*row["detector_bypass"],
                    "detector_bypass_is_conditional_on_target_success":not cfg.smoke,
                    "elapsed_seconds":time.time()-started,
                    "status":"valid_direct_adaptive_evaluation" if validity["passed"] else "invalid_attack_do_not_claim_robustness"})
        atomic_json(row,result_path);mark_complete(result_path);rows[name]=row;aggregate_results(Path(cfg.root))
    clean={"classifier_accuracy":float(torch.tensor(clean_pred).eq(test_y).float().mean()),
            "verifier_accuracy":float(torch.tensor(verifier_pred).eq(test_y).float().mean()),"clean_samples":len(test_x),
            "calibration_samples":len(cal_x),"threshold_source":"clean training calibration split only",
            "dual_model_latency_ms_per_sample":verifier_latency_ms(model,verifier,test_x,device,min(64,cfg.attack_batch_size))}
    return rows,clean


def fit_baselines(cfg,model,train_ds,splits,device,force=False):
    root=cfg.run_dir/"baselines";path=root/"artifact.pt"
    if completed(path,force):return torch.load(path,map_location="cpu",weights_only=False)
    guide_n=effective_per_class(train_ds,splits["guidance"],cfg.guidance_per_class,cfg.n_classes)
    clean_x,clean_y=balanced_tensors(train_ds,splits["guidance"],guide_n,cfg.n_classes)
    pool=torch.load(cfg.run_dir/"guidance/guidance_targeted_pgd.pt",map_location="cpu",weights_only=False)
    clean_acts=activation_matrix(model,clean_x,cfg,device);adv_acts=activation_matrix(model,pool["adversarial"],cfg,device)
    means=[];variances=[]
    for cls in range(cfg.n_classes):
        z=clean_acts[clean_y.eq(cls)];means.append(z.mean(0));variances.append(z.var(0,unbiased=False).clamp_min(1e-5))
    X=torch.cat([clean_acts,adv_acts]);y=np.r_[np.zeros(len(clean_acts),dtype=int),np.ones(len(adv_acts),dtype=int)]
    feature_idx=X.var(0).topk(min(2048,X.shape[1])).indices
    logistic=SGDClassifier(loss="log_loss",penalty="l2",alpha=1e-4,class_weight="balanced",max_iter=3000,tol=1e-4,
                           random_state=cfg.seed,average=True).fit(X[:,feature_idx].numpy(),y)
    artifact={"means":torch.stack(means),"variances":torch.stack(variances),"feature_idx":feature_idx,"logistic":logistic,
              "logistic_coef":torch.tensor(logistic.coef_[0],dtype=torch.float32),"logistic_bias":torch.tensor(float(logistic.intercept_[0])),
              "training_constraint":"same clean guidance split + same successful targeted guidance PGD as proposed method"}
    # Scale only stabilizes the attack optimizer; each reported baseline retains its original score.
    decision=(clean_acts[:,feature_idx].matmul(artifact["logistic_coef"])+artifact["logistic_bias"]).numpy()
    pred=batched_logits(model,clean_x,device).argmax(1).numpy();_,scale=_robust_centers(decision,pred,cfg.n_classes,.25)
    artifact["activation_logistic_decision_scale"]=scale
    for name in ("msp","energy","diagonal_mahalanobis"):
        values=baseline_scores(model,artifact,clean_x,clean_acts,pred,device)[name];_,scale=_robust_centers(values,pred,cfg.n_classes,.02)
        artifact[f"{name}_scale"]=scale
    atomic_torch_save(artifact,path);mark_complete(path);return artifact


def baseline_scores(model,artifact,x,acts,pred,device):
    logits=batched_logits(model,x,device);prob=logits.softmax(1);p=torch.tensor(pred,dtype=torch.long)
    mahal=((acts-artifact["means"][p]).pow(2)/artifact["variances"][p]).mean(1).numpy()
    logistic=artifact["logistic"].predict_proba(acts[:,artifact["feature_idx"]].numpy())[:,1]
    return {"msp":(1-prob.max(1).values).numpy(),"energy":(-torch.logsumexp(logits,1)).numpy(),
            "diagonal_mahalanobis":mahal,"activation_logistic":logistic}


def _baseline_objective(cfg,model,artifact,x,targets,kind,device):
    acts=activation_matrix(model,x,cfg,device);pred=np.asarray(targets);raw=baseline_scores(model,artifact,x,acts,pred,device)[kind]
    if kind=="activation_logistic":
        return (acts[:,artifact["feature_idx"]].matmul(artifact["logistic_coef"])+artifact["logistic_bias"]).numpy()
    return raw


def _baseline_validity(cfg,model,artifact,pool,kind,device,n=64):
    n=min(n,len(pool["adversarial"]));x=pool["original"][:n].to(device);targets=pool["targets"][:n]
    extractor=GraphActivationExtractor(model,cfg.n_last_layers)
    try:ordinary=adaptive_baseline_pgd(model,extractor,x,targets.to(device),cfg.epsilon_linf,cfg.step_linf,
                                       cfg.attack_steps,kind,artifact,(0.,0.,0.,0.)).cpu()
    finally:extractor.close()
    ok=batched_logits(model,ordinary,device).argmax(1).eq(targets)
    if not ok.any():return {"passed":False,"failures":["same-budget classifier-only PGD produced no matched successes"]}
    before=_baseline_objective(cfg,model,artifact,ordinary[ok],targets[ok],kind,device)
    after=_baseline_objective(cfg,model,artifact,pool["adversarial"][:n][ok],targets[ok],kind,device)
    improvement=float(before.mean()-after.mean());standardized=improvement/max(1e-4,float(np.std(before)));fraction=float((after<before).mean())
    failures=[]
    if standardized<.1:failures.append("baseline objective did not improve by 0.1 standard deviations")
    if fraction<.55:failures.append("baseline objective improved on fewer than 55% of matched samples")
    return {"passed":not failures,"failures":failures,"matched_samples":int(ok.sum()),
            "comparator_attack":"same-budget classifier-only Linf PGD","comparator_restarts":4,
            "standardized_objective_improvement":standardized,"fraction_improved":fraction}


def run_baselines(cfg,model,train_ds,test_ds,splits,device,force=False):
    root=cfg.run_dir/"baselines";artifact=fit_baselines(cfg,model,train_ds,splits,device,force)
    cal_x,_,test_x,_=calibration_and_test(cfg,train_ds,test_ds,splits);cal_acts=activation_matrix(model,cal_x,cfg,device);test_acts=activation_matrix(model,test_x,cfg,device)
    cal_pred=batched_logits(model,cal_x,device).argmax(1).numpy();test_pred=batched_logits(model,test_x,device).argmax(1).numpy()
    cal_values=baseline_scores(model,artifact,cal_x,cal_acts,cal_pred,device);clean_values=baseline_scores(model,artifact,test_x,test_acts,test_pred,device)
    loader=DataLoader(test_ds,batch_size=cfg.attack_batch_size,shuffle=False);attack_n=cfg.final_attack_per_target;rows={}
    for kind in BASELINES:
        result_path=root/f"direct_{kind}.json"
        if completed(result_path,force):rows[kind]=json.loads(result_path.read_text());continue
        started=time.time();extractor=GraphActivationExtractor(model,cfg.n_last_layers)
        attack=lambda x,t,k=kind:adaptive_baseline_pgd(model,extractor,x,t,cfg.epsilon_linf,cfg.step_linf,cfg.attack_steps,k,artifact,(.05,.2,1.,5.))
        try:pool=cached_pool(root/f"direct_{kind}.pt",lambda a=attack,k=kind:collect_protocol_pool(
            cfg,a,model,loader,attack_n,cfg.seed+5101,device,f"direct-baseline-{k}"),force)
        finally:extractor.close()
        adv_acts=activation_matrix(model,pool["adversarial"],cfg,device);adv_pred=batched_logits(model,pool["adversarial"],device).argmax(1).numpy()
        adv_values=baseline_scores(model,artifact,pool["adversarial"],adv_acts,adv_pred,device)[kind]
        thresholds=class_conditional_thresholds(cal_values[kind],cal_pred,.05,cfg.n_classes)
        row=metrics_class_conditional(clean_values[kind],test_pred,adv_values,adv_pred,thresholds);validity=_baseline_validity(cfg,model,artifact,pool,kind,device)
        if row["detector_bypass"]>=.9 and not validity["passed"] and not cfg.smoke:
            validity={**validity,"passed":True,"failures":[],"sufficiency_note":"direct attack already bypasses at least 90%"}
        if cfg.smoke:validity={**validity,"passed":False,"failures":[*validity.get("failures",[]),"smoke artifacts are structural tests only"]}
        row.update({"baseline":kind,"evaluation_type":"direct: attack optimized against this baseline's own reported score",
                    "attack_validity":validity,"attack_yield":{k:v for k,v in pool.items() if not torch.is_tensor(v)},
                    "perturbation":perturbation_stats(pool,"linf",cfg.epsilon_linf),
                    "estimated_joint_target_success_and_bypass_rate":pool.get("targeted_asr_on_eligible",pool.get("targeted_asr",0))*row["detector_bypass"],
                    "detector_bypass_is_conditional_on_target_success":not cfg.smoke,"elapsed_seconds":time.time()-started})
        atomic_json(row,result_path);mark_complete(result_path);rows[kind]=row;aggregate_results(Path(cfg.root))
    # Clearly labeled transfer evaluation on every proposed-method attack pool.
    transfer={}
    for attack_name in ATTACKS:
        pool_path=cfg.run_dir/"attacks"/f"{attack_name}.pt"
        if not completed(pool_path,False):continue
        pool=torch.load(pool_path,map_location="cpu",weights_only=False);acts=activation_matrix(model,pool["adversarial"],cfg,device)
        pred=batched_logits(model,pool["adversarial"],device).argmax(1).numpy();values=baseline_scores(model,artifact,pool["adversarial"],acts,pred,device);transfer[attack_name]={}
        for kind in BASELINES:
            thresholds=class_conditional_thresholds(cal_values[kind],cal_pred,.05,cfg.n_classes)
            row=metrics_class_conditional(clean_values[kind],test_pred,values[kind],pred,thresholds)
            row["evaluation_type"]="transfer only: attack optimized against proposed verifier, not baseline";transfer[attack_name][kind]=row
    transfer_path=root/"transfer_on_proposed_attacks.json";atomic_json(transfer,transfer_path);mark_complete(transfer_path)
    return rows,transfer


def run_ablations(cfg,model,geometry,train_ds,test_ds,splits,device,force=False):
    rows={}
    for variant in ABLATIONS:
        verifier,_=train_verifier(cfg,model,geometry,train_ds,splits,device,variant,force)
        attacks,clean=run_verifier_attacks(cfg,model,verifier,train_ds,test_ds,splits,device,force,("joint_pgd_linf",),variant)
        rows[variant]={"clean":clean,"joint_pgd_linf":attacks["joint_pgd_linf"]}
        path=cfg.run_dir/f"ablations/{variant}/result.json";atomic_json(rows[variant],path);mark_complete(path);aggregate_results(Path(cfg.root))
    summary=cfg.run_dir/"ablations/summary.json";atomic_json(rows,summary);mark_complete(summary);return rows


def protocol_description(cfg):
    manifest=study_manifest(cfg.study_profile,cfg.smoke)
    return {"version":"publication_v1_locked","study_manifest":manifest,"main_method":"signature-guided robust dual-model verifier",
            "datasets":manifest["datasets"],"architectures":manifest["architectures"],"seeds":manifest["seeds"],"direct_attacks":list(ATTACKS),
            "training_attack":"short targeted Linf PGD with signature-geometry camouflage",
            "unseen_during_training":["joint_apgd_linf","joint_cw_l2","autoattack_candidate_refine_linf"],
            "baselines":list(BASELINES),"ablations":list(ABLATIONS),
            "partitions":{"classifier":"70% guidance+verifier partitions of official training set",
                          "guidance_verifier_development_calibration":"disjoint stratified partitions; development and calibration unseen by classifier and verifier training",
                          "final":"official test set only"},
            "calibration":"class-conditional thresholds fixed at 5% clean FPR using training calibration partition",
            "classifier_accuracy_gates":MIN_CLASSIFIER_ACCURACY,
            "attack_eligibility":"official-test examples classified correctly before attack; predetermined targets independent of outcome",
            "attack_gate":"same-family/same-norm/same-restart-budget classifier-only control, norm validation, and detector-objective improvement diagnostic",
            "attack_implementations":{"joint_pgd_linf":"project implementation, multi-weight random-restart PGD",
                "joint_apgd_linf":"project APGD-style implementation with checkpointed step reduction",
                "joint_cw_l2":"project tanh-parameterized targeted CW-style L2 implementation",
                "autoattack_candidate_refine_linf":"torchattacks AutoAttack-standard candidate generation followed by targeted detector-aware PGD refinement; not official detector-aware AutoAttack"},
            "configuration":asdict(cfg)}


def run_one(cfg,device,train_missing=False,force=False,component="all",run_ablation_suite=False):
    cfg.run_dir.mkdir(parents=True,exist_ok=True);seed_everything(cfg.seed)
    if train_missing and (component in ("all","classifier") or not completed(cfg.classifier_checkpoint,False)):
        train_classifier(cfg,device,force and component in ("classifier","all"))
    model,classifier_artifact=load_classifier(cfg,device)
    if component=="classifier":return {"classifier":classifier_artifact}
    train_ds=dataset_object(cfg.dataset,True,False);test_ds=dataset_object(cfg.dataset,False,False);splits=detector_splits(train_ds,cfg.seed,cfg.n_classes)
    if component in ("all","guidance") or not completed(cfg.run_dir/"guidance/signature_guidance.pt",False):
        guidance,geometry=train_signature_guidance(cfg,model,train_ds,splits,device,force and component in ("guidance","all"))
    else:
        guidance=torch.load(cfg.run_dir/"guidance/signature_guidance.pt",map_location="cpu",weights_only=False);geometry=geometry_from_artifact(guidance,device)
    if component=="guidance":return {"guidance":guidance}
    if component in ("all","verifier") or not completed(cfg.run_dir/"verifier/verifier.pt",False):
        verifier,verifier_artifact=train_verifier(cfg,model,geometry,train_ds,splits,device,"full",force and component in ("verifier","all"))
    else:verifier,verifier_artifact=_load_verifier(model,cfg.run_dir/"verifier/verifier.pt",device)
    if component=="verifier":return {"verifier":verifier_artifact}
    attacks=clean=baselines=transfer=None
    if component in ("all","attacks"):
        attacks,clean=run_verifier_attacks(cfg,model,verifier,train_ds,test_ds,splits,device,force and component in ("attacks","all"))
    if component in ("all","baselines"):
        baselines,transfer=run_baselines(cfg,model,train_ds,test_ds,splits,device,force and component in ("baselines","all"))
    if component=="ablations" or (component=="all" and run_ablation_suite):run_ablations(cfg,model,geometry,train_ds,test_ds,splits,device,force)
    if component not in ("all","summary"):return {"component":component,"complete":True}
    attack_rows={p.stem:json.loads(p.read_text()) for p in (cfg.run_dir/"attacks").glob("*.json") if completed(p,False)}
    baseline_rows={p.stem.removeprefix("direct_"):json.loads(p.read_text()) for p in (cfg.run_dir/"baselines").glob("direct_*.json") if completed(p,False)}
    if len(attack_rows)!=len(ATTACKS) or len(baseline_rows)!=len(BASELINES):
        raise RuntimeError(f"Run summary refused: attacks={len(attack_rows)}/{len(ATTACKS)}, baselines={len(baseline_rows)}/{len(BASELINES)}")
    if clean is None:
        _,clean=run_verifier_attacks(cfg,model,verifier,train_ds,test_ds,splits,device,False,())
    classifier_summary={k:v for k,v in classifier_artifact.items() if k not in ("state_dict","optimizer","scheduler")}
    main=attack_rows["joint_pgd_linf"];comparisons={}
    for name,row in baseline_rows.items():
        comparisons[name]={"both_attack_gates_passed":bool(main.get("attack_validity",{}).get("passed") and row.get("attack_validity",{}).get("passed")),
            "roc_auc_delta_proposed_minus_baseline":main["roc_auc"]-row["roc_auc"],
            "tpr5_delta_proposed_minus_baseline":main["tpr_at_5pct_fpr"]-row["tpr_at_5pct_fpr"],
            "bypass_delta_proposed_minus_baseline":main["detector_bypass"]-row["detector_bypass"]}
    summary={"protocol":protocol_description(cfg),"classifier":classifier_summary,"verifier_checkpoint":{
        "best_score":verifier_artifact.get("best_score"),"best_epoch":max(verifier_artifact.get("history",[{"epoch":0}]),key=lambda r:r.get("selection_score",-1))["epoch"]},
        "clean":clean,"attacks":attack_rows,"direct_baselines":baseline_rows,
        "comparison_on_direct_joint_pgd_linf":comparisons,
        "all_attack_gates_passed":all(r.get("attack_validity",{}).get("passed",False) for r in attack_rows.values()),
        "completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status":"structural_smoke_only" if cfg.smoke else "complete_protocol_run"}
    path=cfg.run_dir/"run_summary.json";atomic_json(summary,path);mark_complete(path);aggregate_results(Path(cfg.root));return summary


def acceptable_run(cfg,require_ablations=False):
    """Return True only for a complete run whose evaluation validity gates passed."""
    path=cfg.run_dir/"run_summary.json"
    if not completed(path,False):return False
    try:summary=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return False
    attacks=summary.get("attacks",{});baselines=summary.get("direct_baselines",{})
    valid=(summary.get("status")=="complete_protocol_run" and len(attacks)==len(ATTACKS) and
           len(baselines)==len(BASELINES) and
           all(row.get("attack_validity",{}).get("passed",False) for row in attacks.values()) and
           all(row.get("attack_validity",{}).get("passed",False) for row in baselines.values()))
    if not valid or not require_ablations:return valid
    ablation_path=cfg.run_dir/"ablations/summary.json"
    if not completed(ablation_path,False):return False
    try:ablations=json.loads(ablation_path.read_text())
    except (OSError,json.JSONDecodeError):return False
    return len(ablations)==len(ABLATIONS) and all(
        row.get("joint_pgd_linf",{}).get("attack_validity",{}).get("passed",False)
        for row in ablations.values())


def experiment_matrix(datasets,architectures,seeds):
    """Seed-major ordering enforces completion of one replication before the next."""
    return [(seed,dataset,architecture) for seed in seeds for dataset in datasets for architecture in architectures]


def cross_architecture_transfer(base_cfg,dataset,seed,device,force=False):
    """Evaluate saved attacks from every source architecture on every other verifier."""
    rows=[]
    for target_arch in ARCHITECTURES:
        target_cfg=replace(base_cfg,dataset=dataset,architecture=target_arch,seed=seed)
        if not completed(target_cfg.run_dir/"run_summary.json",False):continue
        model,_=load_classifier(target_cfg,device);verifier,_=_load_verifier(model,target_cfg.run_dir/"verifier/verifier.pt",device)
        train_ds=dataset_object(dataset,True,False);splits=detector_splits(train_ds,seed,target_cfg.n_classes)
        cal_n=effective_per_class(train_ds,splits["calibration"],target_cfg.calibration_per_class,target_cfg.n_classes)
        cal_x,_=balanced_tensors(train_ds,splits["calibration"],cal_n,target_cfg.n_classes)
        cal_score,cal_pred,_=verifier_values(model,verifier,cal_x,device);thresholds=class_conditional_thresholds(cal_score,cal_pred,.05,target_cfg.n_classes)
        for source_arch in ARCHITECTURES:
            if source_arch==target_arch:continue
            source_cfg=replace(base_cfg,dataset=dataset,architecture=source_arch,seed=seed)
            for attack in ATTACKS:
                pool_path=source_cfg.run_dir/"attacks"/f"{attack}.pt";out=target_cfg.run_dir/"cross_arch_transfer"/f"from_{source_arch}_{attack}.json"
                if not completed(pool_path,False):continue
                if completed(out,force):rows.append(json.loads(out.read_text()));continue
                pool=torch.load(pool_path,map_location="cpu",weights_only=False);score,pred,_=verifier_values(model,verifier,pool["adversarial"],device)
                targets=pool["targets"].numpy();success=pred==targets;sample_threshold=np.asarray([thresholds[int(c)] for c in pred]);bypass=score<sample_threshold
                row={"dataset":dataset,"seed":seed,"source_architecture":source_arch,"target_architecture":target_arch,"attack":attack,
                     "evaluation_type":"cross-architecture transfer; no optimization against target verifier",
                     "target_classifier_targeted_asr":float(success.mean()),
                     "target_detector_bypass_given_target_success":float(bypass[success].mean()) if success.any() else None,
                     "joint_target_success_and_bypass_rate":float((success&bypass).mean()),"samples":len(score)}
                atomic_json(row,out);mark_complete(out);rows.append(row);aggregate_results(Path(base_cfg.root))
        del model,verifier
        if device.type=="cuda":torch.cuda.empty_cache()
    aggregate_results(Path(base_cfg.root));return rows


def _descriptive(values):
    z=np.asarray(values,dtype=float);n=len(z)
    return {"mean":float(z.mean()),"std":float(z.std(ddof=1)) if n>1 else 0.,"n":n,
            "ci95_half_width":float(student_t.ppf(.975,n-1)*z.std(ddof=1)/np.sqrt(n)) if n>1 else None}


def _artifact_coordinates(path):
    parts=Path(path).parts;seed_pos=next(i for i,x in enumerate(parts) if x.startswith("seed_"))
    return parts[seed_pos-2],parts[seed_pos-1],int(parts[seed_pos].split("_")[1])


def study_manifest(profile="minimum",smoke=False):
    if smoke:
        datasets_expected=["cifar10"];architectures_expected=["resnet18"];seeds_expected=[0]
        ablation_runs=[];cross_rows=0;profile="smoke"
    elif profile=="minimum":
        datasets_expected=list(DATASETS);architectures_expected=["resnet18"];seeds_expected=[0,1]
        ablation_runs=[{"dataset":"cifar10","architecture":"resnet18","seed":0}];cross_rows=0
    else:
        datasets_expected=list(DATASETS);architectures_expected=list(ARCHITECTURES);seeds_expected=[0,1,2]
        ablation_runs=[{"dataset":"cifar10","architecture":"resnet18","seed":s} for s in seeds_expected]
        cross_rows=len(DATASETS)*len(seeds_expected)*len(ARCHITECTURES)*(len(ARCHITECTURES)-1)*len(ATTACKS)
    return {"study_profile":profile,"datasets":datasets_expected,"architectures":architectures_expected,"seeds":seeds_expected,
            "ablation_runs":ablation_runs,"expected_cross_architecture_transfer_rows":cross_rows,
            "scope_note":("minimum thesis study: three datasets, one standard architecture, two seeds, all attacks/baselines, representative ablation"
                          if profile=="minimum" else "full matrix" if profile=="full" else "structural test only")}


def aggregate_results(root=Path("Results/publication_v1")):
    root=Path(root);runs=[];attack_records=[];baseline_records=[];baseline_transfer_records=[];ablation_records=[];transfer_records=[]
    for p in root.glob("**/run_summary.json"):
        if not completed(p,False):continue
        runs.append(json.loads(p.read_text()))
    for p in root.glob("**/attacks/*.json"):
        if "ablations" in p.parts or not completed(p,False):continue
        dataset,architecture,seed=_artifact_coordinates(p);attack_records.append({"dataset":dataset,"architecture":architecture,"seed":seed,"method":p.stem,**json.loads(p.read_text())})
    for p in root.glob("**/baselines/direct_*.json"):
        if not completed(p,False):continue
        dataset,architecture,seed=_artifact_coordinates(p);baseline_records.append({"dataset":dataset,"architecture":architecture,"seed":seed,
            "method":p.stem.removeprefix("direct_"),**json.loads(p.read_text())})
    for p in root.glob("**/baselines/transfer_on_proposed_attacks.json"):
        if not completed(p,False):continue
        dataset,architecture,seed=_artifact_coordinates(p)
        for attack,methods in json.loads(p.read_text()).items():
            for method,row in methods.items():baseline_transfer_records.append({"dataset":dataset,"architecture":architecture,"seed":seed,
                "source_attack":attack,"method":method,**row})
    for p in root.glob("**/ablations/*/result.json"):
        if completed(p,False):
            dataset,architecture,seed=_artifact_coordinates(p);ablation_records.append({"dataset":dataset,"architecture":architecture,"seed":seed,
                "method":p.parent.name,**json.loads(p.read_text())})
    for p in root.glob("**/cross_arch_transfer/*.json"):
        if completed(p,False):transfer_records.append(json.loads(p.read_text()))
    manifest_path=root/"STUDY_MANIFEST.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else study_manifest("full")
    expected=[{"dataset":d,"architecture":a,"seed":s} for d in manifest["datasets"] for a in manifest["architectures"] for s in manifest["seeds"]]
    expected_keys={(x["dataset"],x["architecture"],x["seed"]) for x in expected}
    qualified_runs=[r for r in runs if (r["protocol"]["configuration"]["dataset"],r["protocol"]["configuration"]["architecture"],
                                         r["protocol"]["configuration"]["seed"]) in expected_keys]
    completed_keys={(r["protocol"]["configuration"]["dataset"],r["protocol"]["configuration"]["architecture"],r["protocol"]["configuration"]["seed"]) for r in runs}
    missing=[x for x in expected if (x["dataset"],x["architecture"],x["seed"]) not in completed_keys]
    summaries={}
    for family,records in (("proposed_attacks",attack_records),("direct_baselines",baseline_records)):
        grouped={}
        for row in records:
            key=(row["dataset"],row["architecture"],row["method"]);grouped.setdefault(key,[]).append(row)
        summaries[family]={}
        for key,items in sorted(grouped.items()):
            valid=[r for r in items if r.get("attack_validity",{}).get("passed",False)]
            label="|".join(map(str,key));summaries[family][label]={"completed":len(items),"valid":len(valid)}
            for metric in ("roc_auc","tpr_at_1pct_fpr","tpr_at_5pct_fpr","clean_fpr","detector_bypass","estimated_joint_target_success_and_bypass_rate"):
                values=[r[metric] for r in valid if isinstance(r.get(metric),(int,float))]
                if values:summaries[family][label][metric]=_descriptive(values)
    q_attacks=[r for r in attack_records if (r["dataset"],r["architecture"],r["seed"]) in expected_keys]
    q_baselines=[r for r in baseline_records if (r["dataset"],r["architecture"],r["seed"]) in expected_keys]
    q_baseline_transfer=[r for r in baseline_transfer_records if (r["dataset"],r["architecture"],r["seed"]) in expected_keys]
    ablation_keys={(r["dataset"],r["architecture"],r["seed"]) for r in manifest["ablation_runs"]}
    q_ablations=[r for r in ablation_records if (r["dataset"],r["architecture"],r["seed"]) in ablation_keys]
    q_transfer=([] if int(manifest["expected_cross_architecture_transfer_rows"])==0 else [r for r in transfer_records
        if r.get("dataset") in manifest["datasets"] and r.get("seed") in manifest["seeds"] and
           r.get("source_architecture") in manifest["architectures"] and r.get("target_architecture") in manifest["architectures"]])
    expected_counts={"primary_runs":len(expected),"proposed_direct_attacks":len(expected)*len(ATTACKS),
                     "direct_baselines":len(expected)*len(BASELINES),"baseline_transfer_rows":len(expected)*len(ATTACKS)*len(BASELINES),
                     "ablations":len(ablation_keys)*len(ABLATIONS),
                     "cross_architecture_transfer_rows":int(manifest["expected_cross_architecture_transfer_rows"])}
    all_gates=len(q_attacks)==expected_counts["proposed_direct_attacks"] and all(r.get("attack_validity",{}).get("passed",False) for r in q_attacks)
    baseline_gates=len(q_baselines)==expected_counts["direct_baselines"] and all(r.get("attack_validity",{}).get("passed",False) for r in q_baselines)
    ablation_gates=len(q_ablations)==expected_counts["ablations"] and all(
        r.get("joint_pgd_linf",{}).get("attack_validity",{}).get("passed",False) for r in q_ablations)
    artifact_matrix_complete=(len(qualified_runs)==expected_counts["primary_runs"] and len(q_attacks)==expected_counts["proposed_direct_attacks"]
        and len(q_baselines)==expected_counts["direct_baselines"] and len(q_baseline_transfer)==expected_counts["baseline_transfer_rows"]
        and len(q_ablations)==expected_counts["ablations"] and len(q_transfer)==expected_counts["cross_architecture_transfer_rows"])
    failure_path=root/"FAILED_RUNS.json";failed_runs=json.loads(failure_path.read_text()) if failure_path.exists() else []
    aggregate={"protocol_version":"publication_v1_locked","study_manifest":manifest,"expected_runs":len(expected),
               "completed_runs":len(qualified_runs),"missing_runs":missing,
                "expected_artifact_counts":expected_counts,"publication_matrix_complete":artifact_matrix_complete,
               "all_proposed_attack_gates_passed":all_gates,"all_direct_baseline_gates_passed":baseline_gates,
               "all_ablation_attack_gates_passed":ablation_gates,
               "publication_qualified":bool(artifact_matrix_complete and all_gates and baseline_gates and ablation_gates),
               "failed_runs":failed_runs,"runs":runs,"attack_records":attack_records,
               "baseline_records":baseline_records,"baseline_transfer_records":baseline_transfer_records,"ablation_records":ablation_records,
               "cross_architecture_transfer":transfer_records,
               "descriptive_summaries":summaries,"generated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json_path=root/"AGGREGATE_RESULTS.json";atomic_json(aggregate,json_path);mark_complete(json_path)
    lines=["SIGNATURE-GUIDED VERIFIER - PUBLICATION V1",f"Study profile: {manifest['study_profile']}",manifest["scope_note"],
           f"Completed runs: {len(qualified_runs)}/{len(expected)}",
           f"Failed combinations: {len(failed_runs)}",f"Artifact matrix complete: {artifact_matrix_complete}",
           f"All proposed attack gates passed: {all_gates}",f"All direct baseline gates passed: {baseline_gates}",
           f"All ablation attack gates passed: {ablation_gates}",
           f"Publication qualified: {aggregate['publication_qualified']}",""]
    if missing:
        lines.append("MISSING RUNS")
        lines.extend(f"  {x['dataset']} | {x['architecture']} | seed {x['seed']}" for x in missing);lines.append("")
    lines.append(f"Completed direct attack artifacts: {len(attack_records)}")
    lines.append(f"Completed direct baseline artifacts: {len(baseline_records)}")
    lines.append(f"Completed ablation artifacts: {len(ablation_records)}")
    lines.append(f"Completed baseline-transfer rows: {len(baseline_transfer_records)}")
    lines.append(f"Completed cross-architecture-transfer rows: {len(transfer_records)}\n")
    lines.append("VALID DIRECT ADAPTIVE RESULTS (MEAN +/- STD)")
    for family,groups in summaries.items():
        lines.append(f"[{family}]")
        for name,data in groups.items():
            auc=data.get("roc_auc");bypass=data.get("detector_bypass");tpr=data.get("tpr_at_5pct_fpr")
            if auc:lines.append(f"{name}: AUC={auc['mean']:.4f}+/-{auc['std']:.4f}; TPR5={tpr['mean']:.4f}+/-{tpr['std']:.4f}; bypass={bypass['mean']:.4f}+/-{bypass['std']:.4f}; n={auc['n']}")
            else:lines.append(f"{name}: no valid completed attack")
    text_path=root/"AGGREGATE_RESULTS.txt";text_path.write_text("\n".join(lines),encoding="utf-8")
    return json_path


def _csv_values(value,allowed):
    values=[x.strip() for x in value.split(",") if x.strip()]
    if values==["all"]:return list(allowed)
    invalid=set(values)-set(map(str,allowed))
    if invalid:raise ValueError(f"Unsupported values: {sorted(invalid)}")
    return values


def main(argv=None):
    p=argparse.ArgumentParser(description="Locked publication matrix for the signature-guided robust verifier")
    p.add_argument("--datasets",default="all",help="Comma-separated or all: cifar10,svhn,stl10")
    p.add_argument("--architectures",default="all",help="Comma-separated or all: resnet18,densenet121,convnext_tiny")
    p.add_argument("--seeds",default="0,1",help="Comma-separated integer seeds; completed valid runs are skipped")
    p.add_argument("--component",choices=["all","classifier","guidance","verifier","attacks","baselines","ablations","summary"],default="all")
    p.add_argument("--train-missing-classifiers",action="store_true",help="Explicitly authorize training missing classifier checkpoints")
    p.add_argument("--skip-ablations",action="store_true",help="Skip profile ablations (minimum: CIFAR-10/ResNet-18/seed 0)")
    p.add_argument("--skip-cross-architecture-transfer",action="store_true")
    p.add_argument("--study-profile",choices=["minimum","full"],default="minimum",
                   help="minimum: 3 datasets x ResNet-18 x 2 seeds, representative ablation; full: original 27-run matrix")
    p.add_argument("--force",action="store_true",help="Rerun the selected component even if complete")
    p.add_argument("--smoke",action="store_true",help="Tiny end-to-end structural test; uses a separate result root")
    p.add_argument("--classifier-epochs",type=int,default=50);p.add_argument("--verifier-epochs",type=int,default=20)
    p.add_argument("--attack-steps",type=int,default=80);p.add_argument("--final-attack-per-target",type=int,default=100)
    p.add_argument("--verifier-batch-size",type=int,default=32);p.add_argument("--attack-batch-size",type=int,default=32)
    p.add_argument("--device",default="auto",help="Execution device: auto, cpu, cuda, cuda:0, cuda:1, ...")
    a=p.parse_args(argv);datasets_selected=_csv_values(a.datasets,DATASETS);architectures_selected=_csv_values(a.architectures,ARCHITECTURES)
    seeds=[int(x) for x in a.seeds.split(",")]
    if a.study_profile=="minimum" and a.architectures=="all":architectures_selected=["resnet18"]
    if not a.smoke and a.component in ("all","attacks") and importlib.util.find_spec("torchattacks") is None:
        p.error("torchattacks is required for AutoAttack candidate/refinement; install requirements.txt before attack evaluation")
    if a.smoke and a.datasets=="all" and a.architectures=="all" and a.seeds=="0,1":datasets_selected=["cifar10"];architectures_selected=["resnet18"];seeds=[0]
    device=resolve_device(a.device);base=PublicationConfig(classifier_epochs=a.classifier_epochs,
        verifier_epochs=a.verifier_epochs,attack_steps=a.attack_steps,final_attack_per_target=a.final_attack_per_target,
        verifier_batch_size=a.verifier_batch_size,attack_batch_size=a.attack_batch_size,study_profile=a.study_profile)
    if a.smoke:base=smoke_config(base)
    manifest=study_manifest(a.study_profile,a.smoke);manifest_path=Path(base.root)/"STUDY_MANIFEST.json"
    atomic_json(manifest,manifest_path);mark_complete(manifest_path)
    failures=[]
    for seed in seeds:
        seed_failures=[]
        for dataset in datasets_selected:
            for architecture in architectures_selected:
                cfg=resource_adjusted_config(replace(base,dataset=dataset,architecture=architecture,seed=seed));print(f"\n=== {dataset} | {architecture} | seed {seed} | {a.component} ===",flush=True)
                ablate=(not a.skip_ablations and dataset=="cifar10" and architecture=="resnet18" and
                        (a.study_profile=="full" or seed==0))
                if a.component=="ablations" and not ablate:
                    print("[skip] Ablations are outside the selected study profile for this combination",flush=True);continue
                if a.component=="all" and not a.force and acceptable_run(cfg,ablate):
                    print("[skip acceptable] Complete artifacts and all validity gates passed",flush=True);continue
                try:
                    run_one(cfg,device,a.train_missing_classifiers,a.force,a.component,ablate)
                except Exception as exc:
                    initial_error=repr(exc);exc.__traceback__=None
                    if is_cuda_oom(exc) and device.type=="cuda":
                        retry_cfg=replace(cfg,verifier_batch_size=max(4,cfg.verifier_batch_size//2),
                                          attack_batch_size=max(2,cfg.attack_batch_size//2))
                        print(f"[OOM] Retrying from the last completed artifact with verifier batch "
                              f"{cfg.verifier_batch_size}->{retry_cfg.verifier_batch_size}, attack batch "
                              f"{cfg.attack_batch_size}->{retry_cfg.attack_batch_size}",flush=True)
                        safe_cuda_cleanup()
                        try:run_one(retry_cfg,device,a.train_missing_classifiers,False,a.component,ablate)
                        except Exception as retry_exc:
                            retry_error=repr(retry_exc);retry_exc.__traceback__=None
                            failure={"dataset":dataset,"architecture":architecture,"seed":seed,
                                "error":retry_error,"initial_error":initial_error,"automatic_oom_retry":True}
                            failures.append(failure);seed_failures.append(failure);print(f"[FAILED] {failure}",flush=True)
                    else:
                        failure={"dataset":dataset,"architecture":architecture,"seed":seed,"error":initial_error}
                        failures.append(failure);seed_failures.append(failure);print(f"[FAILED] {failure}",flush=True)
                finally:
                    if device.type=="cuda":safe_cuda_cleanup()
        if a.component=="all":
            unacceptable=[]
            for dataset in datasets_selected:
                for architecture in architectures_selected:
                    cfg=resource_adjusted_config(replace(base,dataset=dataset,architecture=architecture,seed=seed))
                    ablate=(not a.skip_ablations and dataset=="cifar10" and architecture=="resnet18" and
                            (a.study_profile=="full" or seed==0))
                    if not acceptable_run(cfg,ablate):unacceptable.append({"dataset":dataset,"architecture":architecture,"seed":seed})
            seed_status={"seed":seed,"complete_and_acceptable":not seed_failures and not unacceptable,
                         "unacceptable_or_missing_runs":unacceptable,"failures":seed_failures,
                         "updated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            seed_path=Path(base.root)/f"SEED_{seed}_STATUS.json";atomic_json(seed_status,seed_path);mark_complete(seed_path)
            aggregate_results(Path(base.root))
            if unacceptable or seed_failures:
                failure={"seed":seed,"component":"seed_barrier","error":"Seed is incomplete or failed validity gates; later seeds were not started",
                         "unacceptable_or_missing_runs":unacceptable}
                failures.append(failure);print(f"[SEED BARRIER] {failure}",flush=True);break
        if a.study_profile=="full" and not a.skip_cross_architecture_transfer and set(architectures_selected)==set(ARCHITECTURES):
            for dataset in datasets_selected:
                try:cross_architecture_transfer(base,dataset,seed,device,a.force and a.component=="all")
                except Exception as exc:failures.append({"dataset":dataset,"seed":seed,"component":"transfer","error":repr(exc)})
    failure_path=Path(base.root)/"FAILED_RUNS.json";atomic_json(failures,failure_path)
    aggregate=aggregate_results(Path(base.root));print(f"Aggregate results: {aggregate.resolve()}",flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed; details saved to {failure_path}")


if __name__=="__main__":main()
