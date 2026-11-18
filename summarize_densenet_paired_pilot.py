import argparse,json,torch
from pathlib import Path
from dataclasses import replace
from revision.publication import (PublicationConfig,resource_adjusted_config,load_classifier,_load_verifier,
 dataset_object,detector_splits,calibration_and_test,verifier_values,resolve_device)
from revision.core import class_conditional_thresholds,metrics_class_conditional,atomic_json,mark_complete
cfg=resource_adjusted_config(replace(PublicationConfig(),dataset='cifar10',architecture='densenet121',seed=0))
root=cfg.run_dir
control_path=root/'reviewer_densenet_control.json'
if not control_path.with_suffix('.json.complete').exists():raise RuntimeError('No-signature pilot incomplete')
control=json.loads(control_path.read_text())
full_pool=torch.load(root/'attacks/joint_pgd_linf.pt',map_location='cpu',weights_only=False)
no_pool=torch.load(root/'ablations/no_signature/attacks/joint_pgd_linf.pt',map_location='cpu',weights_only=False)
selected=[]
for c in range(cfg.n_classes):
 idx=torch.where(full_pool['targets'].eq(c))[0][:20]
 if len(idx)!=20:raise RuntimeError(f'Full pool lacks class {c}')
 selected.extend(idx.tolist())
orig_full=full_pool['original'][selected]
matched=torch.equal(orig_full,no_pool['original']) and torch.equal(full_pool['targets'][selected],no_pool['targets'])
print('originals exactly matched',matched,flush=True)
if not matched:print('WARNING: balanced 200-example cohorts differ; report as nonpaired.',flush=True)
parser=argparse.ArgumentParser()
parser.add_argument('--device',default='auto')
device=resolve_device(parser.parse_args().device);model,_=load_classifier(cfg,device)
verifier,_=_load_verifier(model,root/'verifier/verifier.pt',device)
train=dataset_object('cifar10',True,False);test=dataset_object('cifar10',False,False)
splits=detector_splits(train,0,cfg.n_classes)
cal_x,_,test_x,_=calibration_and_test(cfg,train,test,splits)
cal_score,cal_pred,_=verifier_values(model,verifier,cal_x,device)
clean_score,clean_pred,_=verifier_values(model,verifier,test_x,device)
adv_score,adv_pred,_=verifier_values(model,verifier,full_pool['adversarial'][selected],device)
thresholds=class_conditional_thresholds(cal_score,cal_pred,.05,cfg.n_classes)
full=metrics_class_conditional(clean_score,clean_pred,adv_score,adv_pred,thresholds)
no=control['attacks']['joint_pgd_linf']
result={'dataset':'cifar10','architecture':'densenet121','seed':0,'attack':'joint_pgd_linf',
 'samples':200,'per_target':20,'source_inputs_and_targets_identical':matched,
 'comparison_type':'paired inputs' if matched else 'separately attacked class-balanced cohorts',
 'full':{k:full[k] for k in ('roc_auc','tpr_at_5pct_fpr','detector_bypass','clean_fpr')},
 'no_signature':{k:no[k] for k in ('roc_auc','tpr_at_5pct_fpr','detector_bypass','clean_fpr')},
 'no_signature_attack_validity':no['attack_validity']}
out=root/'reviewer_densenet_paired_pgd_200.json';atomic_json(result,out);mark_complete(out)
print(json.dumps(result,indent=2),flush=True)
