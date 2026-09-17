from __future__ import annotations
import copy
import hashlib
import json
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from configuration import parse_args
from data import (METADATA_SCHEMA_VERSION, PairedINbreastDataset, assign_patient_folds, build_metadata,
                  grouped_development_split, load_metadata_pairs, split_audit, _view_from_name, _patient_from_name,
                  _column, _read_metadata, acquisition_key, image_key)
from environment import environment_info
from evaluate import (METRIC_NAMES, apply_calibration, classification_metrics, classification_metrics_from_predictions,
                      dice_iou, fit_temperature, localization_summary, patient_cluster_bootstrap, select_threshold)
from gradcam import paired_gradcam, save_prediction_artifacts
from model import ARCHITECTURE_NAME, ARCHITECTURE_VERSION, NORMALIZATION, SUPPORTED_BACKBONES, create_model
from preprocessing import (PREPROCESSING_VERSION, geometry_valid_region, heatmap_to_original, load_xml_mask, read_dicom)
from reporting import prediction_artifacts, write_json

LABEL_DEFINITION = 'BI-RADS 1-3=0; BI-RADS 4-6=1; higher projection label per breast'


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_seed(worker_id):
    value = torch.initial_seed() % 2**32
    np.random.seed(value); random.seed(value)


def make_loader(frame, args, training, batch_size=None):
    if frame.empty:
        raise ValueError('DataLoader skup je prazan.')
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(PairedINbreastDataset(frame, args.size, training and getattr(args,'augmentations',True),
                      cache_dir=getattr(args,'cache_dir',None)), batch_size=batch_size or args.batch_size,
        shuffle=training, num_workers=args.workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_seed,
        generator=generator, persistent_workers=False)


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, training):
    model.train(training)
    total, labels, probabilities = 0., [], []
    for batch in loader:
        cc,mlo,targets = batch['cc_image'].to(device),batch['mlo_image'].to(device),batch['label'].to(device)
        if not torch.isfinite(targets).all() or not torch.isin(targets,targets.new_tensor([0.,1.])).all():
            raise ValueError('Trening/evaluacija zahteva binarne konačne labele.')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            logits=model(cc,mlo)
            if logits.shape!=targets.shape or not torch.isfinite(logits).all():
                raise ValueError('Model mora vratiti jedan konačan logit po paru.')
            loss=loss_fn(logits,targets)
            if not torch.isfinite(loss):
                raise ValueError('Training loss sadrži NaN/Inf.')
        if training:
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            # CUDA AMP may legitimately overflow early; GradScaler must be allowed
            # to skip that optimizer step and reduce its scale.
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer);scaler.update()
        total+=loss.item()*len(targets)
        labels.extend(targets.cpu().int().tolist())
        probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return total/len(loader.dataset),labels,probabilities


def predict_frame(model,frame,args,device):
    loader=make_loader(frame,args,False)
    by_id=frame.set_index('pair_id').to_dict('index')
    rows=[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits=model(batch['cc_image'].to(device),batch['mlo_image'].to(device))
            if logits.shape!=(len(batch['pair_id']),) or not torch.isfinite(logits).all():
                raise ValueError('Inference mora vratiti jedan konačan logit po paru.')
            probabilities=torch.sigmoid(logits).cpu().tolist()
            for index,probability in enumerate(probabilities):
                pair_id=batch['pair_id'][index]
                metadata=by_id[pair_id]
                label=float(batch['label'][index])
                row={key:(value.item() if isinstance(value,np.generic) else value) for key,value in metadata.items()}
                row.update(pair_id=pair_id,patient_id=batch['patient_id'][index],side=batch['side'][index],
                           true_label=int(label) if np.isfinite(label) else None,
                           probability=float(probability),raw_probability=float(probability),raw_logit=float(logits[index].cpu()))
                rows.append(row)
    return rows


def apply_decisions(rows,checkpoint,fold=0):
    raw=[row.get('raw_probability',row['probability']) for row in rows]
    z=[row['raw_logit'] for row in rows] if all('raw_logit' in row for row in rows) else None
    calibrated=apply_calibration(raw,checkpoint.get('calibration'),logits=z)
    threshold=float(checkpoint.get('threshold',.5))
    for row,p,r in zip(rows,calibrated,raw):
        row.update(raw_probability=float(r),probability=float(p),prediction=int(p>=threshold),threshold=threshold,fold=fold,
                   probability_kind='calibrated' if checkpoint.get('calibration',{}).get('applied') else 'raw',
                   predicted_at_05=int(p>=.5),tuned_threshold=threshold,predicted_tuned=int(p>=threshold),label=row['true_label'])
        if fold:
            row['prepared_holdout_split'] = row.get('split')
            row['split'] = 'test'
            row['evaluation_role'] = 'outer_test'
    return rows


def frame_hash(frame):
    ordered=frame.sort_values('pair_id').reindex(sorted(frame.columns),axis=1)
    return hashlib.sha256(ordered.to_csv(index=False,lineterminator='\n').encode()).hexdigest()


def trusted_checkpoint_load(path):
    warnings.warn(f'Checkpoint se učitava pomoću torch.load(weights_only=False); pickle može izvršiti kod. '
                  f'Koristite isključivo pouzdane lokalne fajlove: {path}', RuntimeWarning)
    return torch.load(path,map_location='cpu',weights_only=False)


def validate_checkpoint_versions(checkpoint):
    if checkpoint.get('preprocessing_version')!=PREPROCESSING_VERSION:
        raise ValueError('Checkpoint koristi staru ili nepoznatu verziju preprocessinga; trenirajte od početka.')
    if checkpoint.get('metadata_schema_version')!=METADATA_SCHEMA_VERSION:
        raise ValueError('Checkpoint koristi nekompatibilnu metadata semu; trenirajte od početka.')
    if checkpoint.get('architecture_name')!=ARCHITECTURE_NAME or checkpoint.get('architecture_version')!=ARCHITECTURE_VERSION:
        raise ValueError('Checkpoint koristi nepoznatu/nekompatibilnu arhitekturu.')
    if checkpoint.get('normalization')!=NORMALIZATION or checkpoint.get('label_definition')!=LABEL_DEFINITION or checkpoint.get('num_outputs')!=1:
        raise ValueError('Checkpoint normalization/label definition/num_outputs nisu kompatibilni.')
    if not isinstance(checkpoint.get('size'),int) or checkpoint['size']<32 or checkpoint['size']%32:
        raise ValueError('Checkpoint input size nije kompatibilan sa modelom.')
    if not 0<=checkpoint.get('dropout',-1)<1 or checkpoint.get('backbone') not in SUPPORTED_BACKBONES:
        raise ValueError('Checkpoint backbone/dropout nisu validni.')
    if not np.isfinite(checkpoint.get('threshold',np.nan)) or not 0<=checkpoint['threshold']<=1:
        raise ValueError('Checkpoint threshold nije validan.')
    apply_calibration([.5],checkpoint.get('calibration'))
    roles = checkpoint.get('patient_roles')
    if not isinstance(roles, dict) or not {'train', 'val', 'threshold'} <= set(roles):
        raise ValueError('Checkpoint nema proverljiv patient grouping provenance.')
    names = sorted(roles)
    if any(set(roles[a]) & set(roles[b]) for i, a in enumerate(names) for b in names[i + 1:]):
        raise ValueError('Checkpoint patient roles sadrže leakage.')


def load_checkpoint(args,device):
    checkpoint=trusted_checkpoint_load(args.checkpoint or args.output_dir/'best.pt')
    validate_checkpoint_versions(checkpoint)
    model=create_model(checkpoint['size'],False,checkpoint['dropout'],checkpoint['backbone']).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    args.size=checkpoint['size']
    return model,checkpoint


def _checkpoint_state(model,optimizer,scheduler,scaler,train_loader,val_loader,epoch,best,stale,history,args,pairs):
    roles={role:sorted(frame.patient_id.astype(str).unique()) for role,frame in pairs.groupby('split')}
    return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,
        'scaler':scaler.state_dict(),'epoch':epoch,'best_metric':best,'best_f1_at_05':best if args.checkpoint_metric=='f1' else None,
        'stale':stale,'history':copy.deepcopy(history),'size':args.size,'dropout':args.dropout,'backbone':args.backbone,
        'num_outputs':1,'normalization':NORMALIZATION,'label_definition':LABEL_DEFINITION,
        'architecture_name':ARCHITECTURE_NAME,'architecture_version':ARCHITECTURE_VERSION,
        'preprocessing_version':PREPROCESSING_VERSION,'metadata_schema_version':METADATA_SCHEMA_VERSION,
        'threshold':args.fixed_threshold,'threshold_selection':{'strategy':'fixed','source':'unfinalized_configuration'},
        'calibration':{'method':'none','applied':False,'probability_kind':'raw'},'postprocessing_finalized':False,
        'config':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        'metadata_hash':frame_hash(pairs),'patient_roles':roles,'environment':environment_info(),
        'device':str(next(model.parameters()).device),'checkpoint_metric':args.checkpoint_metric,
        'train_loader_generator_state':train_loader.generator.get_state(),
        'val_loader_generator_state':val_loader.generator.get_state(),
        'rng_state':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                     'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     'mps':torch.mps.get_rng_state() if next(model.parameters()).device.type == 'mps' else None}}


def make_optimizer(model,args):
    """Use a conservative encoder LR and a separate classifier-head LR."""
    backbone=list(model.backbone.parameters()) if hasattr(model,'backbone') else []
    backbone_ids={id(parameter) for parameter in backbone}
    head=[parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    groups=[]
    if backbone:
        groups.append({'params':backbone,'lr':args.encoder_lr,'name':'encoder'})
    if head:
        groups.append({'params':head,'lr':args.head_lr,'name':'head'})
    return torch.optim.AdamW(groups,weight_decay=args.weight_decay)


def finalize_checkpoint(path,pairs,args,device,model):
    checkpoint=trusted_checkpoint_load(path)
    model.load_state_dict(checkpoint['model']);model.eval()
    calibration={'method':'none','applied':False,'probability_kind':'raw','source':None}
    if args.calibration_method=='temperature':
        calibration_frame=pairs[pairs.split=='calibration']
        rows=predict_frame(model,calibration_frame,args,device)
        calibration=fit_temperature([r['true_label'] for r in rows],[r['probability'] for r in rows],
                                    source='independent_calibration_holdout',bins=args.calibration_bins,logits=[r['raw_logit'] for r in rows])
        calibration['patient_ids_sha256']=split_audit(calibration_frame)['splits']['calibration']['patient_ids_sha256']
    threshold_frame=pairs[pairs.split=='threshold']
    rows=predict_frame(model,threshold_frame,args,device)
    probabilities=apply_calibration([r['probability'] for r in rows],calibration,logits=[r['raw_logit'] for r in rows])
    selection=select_threshold([r['true_label'] for r in rows],probabilities,args.threshold_strategy,args.fixed_threshold,
                               args.minimum_sensitivity,source='independent_threshold_holdout')
    selection['patient_ids_sha256']=split_audit(threshold_frame)['splits']['threshold']['patient_ids_sha256']
    checkpoint.update(threshold=selection['threshold'],threshold_selection=selection,calibration=calibration,postprocessing_finalized=True)
    torch.save(checkpoint,path)
    return checkpoint


def train_model(pairs,args,device):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    audit=split_audit(pairs)
    required={'train','val','threshold'}|({'calibration'} if args.calibration_method!='none' else set())
    if required-set(pairs.split):
        raise ValueError(f'Nedostaju nezavisni development splitovi {required-set(pairs.split)}. Pokrenite prepare sa novom konfiguracijom.')
    for role in required:
        if set(pd.to_numeric(pairs[pairs.split==role].label))!={0,1}:
            raise ValueError(f'{role} skup mora sadržati obe breast-level klase.')
    print(json.dumps(audit,indent=2))
    write_json(args.output_dir/'split_audit.json',audit)
    train_frame,val_frame=pairs[pairs.split=='train'],pairs[pairs.split=='val']
    train_loader,val_loader=make_loader(train_frame,args,True),make_loader(val_frame,args,False)
    model=create_model(args.size,pretrained=not args.no_pretrained and not args.resume,dropout=args.dropout,backbone=args.backbone).to(device)
    model.freeze_backbone(args.freeze_epochs>0)
    positives,negatives=int(train_frame.label.sum()),int((train_frame.label==0).sum())
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives/positives,device=device))
    optimizer=make_optimizer(model,args)
    mode='min' if args.checkpoint_metric=='val_loss' else 'max'
    scheduler=None
    if args.scheduler=='plateau':
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode=mode,factor=.5,patience=2)
    elif args.scheduler=='cosine':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,args.epochs))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=float('inf') if mode=='min' else -float('inf')
    stale,history,first_epoch=0,[],1
    if args.resume:
        saved=trusted_checkpoint_load(args.resume)
        validate_checkpoint_versions(saved)
        if saved.get('metadata_hash')!=frame_hash(pairs):
            raise ValueError('Resume metadata hash/split nije isti kao u checkpointu.')
        existing_history = args.output_dir / 'history.csv'
        if existing_history.is_file():
            previous_history = pd.read_csv(existing_history)
            if not previous_history.empty and previous_history.epoch.max() > saved['epoch']:
                raise ValueError('Resume checkpoint je stariji od postojeće istorije; koristite last.pt ili zaseban output directory.')
        for name in ('size','dropout','backbone','checkpoint_metric','scheduler','freeze_epochs','batch_size','workers','seed',
                     'lr','encoder_lr','head_lr','weight_decay','augmentations','patience','threshold_strategy','fixed_threshold','minimum_sensitivity','calibration_method'):
            if saved['config'].get(name)!=getattr(args,name):
                raise ValueError(f'Resume konfiguracija nije kompatibilna: {name}.')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        if scheduler:
            scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        train_loader.generator.set_state(saved['train_loader_generator_state'].cpu())
        val_loader.generator.set_state(saved['val_loader_generator_state'].cpu())
        best=float(saved['best_metric']);stale=int(saved['stale']);history=list(saved['history']);first_epoch=int(saved['epoch'])+1
        rng=saved['rng_state']
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu())
        if torch.cuda.is_available() and rng.get('cuda') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
        if device.type == 'mps' and rng.get('mps') is not None:
            torch.mps.set_rng_state(rng['mps'].cpu())
        model.freeze_backbone(first_epoch<=args.freeze_epochs)
        source=Path(args.resume).parent
        best_source=source/'best.pt'
        previous=trusted_checkpoint_load(best_source) if best_source.is_file() else None
        if previous is not None:
            validate_checkpoint_versions(previous)
            if previous['metadata_hash']!=frame_hash(pairs):
                raise ValueError('Resume best.pt metadata hash mismatch.')
        coherent=previous is not None and previous['epoch']<=saved['epoch'] and previous['best_metric']==saved['best_metric']
        if not coherent:
            scores=saved['history']
            historical_best=(min(scores,key=lambda row:row['checkpoint_score']) if mode=='min'
                             else max(scores,key=lambda row:row['checkpoint_score']))
            if historical_best['epoch']!=saved['epoch']:
                raise ValueError('Nedostaje best.pt iz vremena resume checkpointa; sačuvajte odgovarajući best.pt ili koristite najnoviji last.pt.')
            previous=saved
        target_best=args.output_dir/'best.pt'
        if not coherent or best_source.resolve()!=target_best.resolve():
            torch.save(previous,target_best)
        target_last=args.output_dir/'last.pt'
        if Path(args.resume).resolve()!=target_last.resolve():
            torch.save(saved,target_last)
        print(f'Nastavak treninga od epohe {first_epoch}: {args.resume}')
    configuration={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    write_json(args.output_dir/'experiment.json',{'config':configuration,'seed':args.seed,'device':str(device),
               'environment':environment_info(),'metadata_hash':frame_hash(pairs),'metadata_schema_version':METADATA_SCHEMA_VERSION,
               'preprocessing_version':PREPROCESSING_VERSION,'architecture_version':ARCHITECTURE_VERSION,
               'checkpoint_retention':f'best.pt, last.pt, plus every {args.checkpoint_every} epochs (0 disables periodic snapshots)'})
    (args.output_dir/'experiment_config.yaml').write_text(yaml.safe_dump({k:v for k,v in configuration.items() if k!='config'},sort_keys=False),encoding='utf-8')
    if first_epoch>args.epochs:
        print(f'Checkpoint je već na epohi {first_epoch-1}; traženo je {args.epochs}. Istorija je očuvana.')
        if history:
            save_history(history,args.output_dir)
        return
    if stale>=args.patience and first_epoch>args.freeze_epochs+1:
        print('Resume checkpoint je već završio early stopping; istorija je očuvana.')
        save_history(history,args.output_dir)
        return
    for epoch in range(first_epoch,args.epochs+1):
        if epoch==args.freeze_epochs+1:
            model.freeze_backbone(False)
            # Give the newly unfrozen encoder a full early-stopping window.
            stale=0
            if args.scheduler=='plateau':
                for group in optimizer.param_groups:
                    group['lr']=args.encoder_lr if group.get('name')=='encoder' else args.head_lr
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,mode=mode,factor=.5,patience=2)
        train_loss,_,_=run_epoch(model,train_loader,loss_fn,optimizer,scaler,device,True)
        val_loss,labels,probabilities=run_epoch(model,val_loader,loss_fn,optimizer,scaler,device,False)
        fixed=classification_metrics(labels,probabilities,.5,args.calibration_bins)
        score=val_loss if args.checkpoint_metric=='val_loss' else fixed[args.checkpoint_metric]
        if score is None or not np.isfinite(score):
            raise ValueError(f'Checkpoint metrika {args.checkpoint_metric} nije definisana.')
        if scheduler:
            scheduler.step(score) if args.scheduler=='plateau' else scheduler.step()
        row={'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss,'f1_at_05':fixed['f1'],
             'sensitivity':fixed['sensitivity'],'roc_auc':fixed['roc_auc'],'pr_auc':fixed['pr_auc'],
             'checkpoint_metric':args.checkpoint_metric,'checkpoint_score':score,
             'lr':optimizer.param_groups[0]['lr'],
             'encoder_lr':next((group['lr'] for group in optimizer.param_groups if group.get('name')=='encoder'),None),
             'head_lr':next((group['lr'] for group in optimizer.param_groups if group.get('name')=='head'),None)}
        history.append(row)
        improved=score<best if mode=='min' else score>best
        best,stale=(score,0) if improved else (best,stale+1)
        state=_checkpoint_state(model,optimizer,scheduler,scaler,train_loader,val_loader,epoch,best,stale,history,args,pairs)
        state['metrics_at_05']=fixed
        torch.save(state,args.output_dir/'last.pt')
        if improved:
            torch.save(state,args.output_dir/'best.pt')
        if args.checkpoint_every and epoch%args.checkpoint_every==0:
            torch.save(state,args.output_dir/f'epoch_{epoch:03d}.pt')
        save_history(history,args.output_dir)
        print(f'Epoha {epoch:03d}/{args.epochs:03d} | train loss={train_loss:.4f} | val loss={val_loss:.4f} | '
              f'{args.checkpoint_metric}={score:.4f} | F1@0.5={fixed["f1"]:.4f} | '
              f'encoder LR={row["encoder_lr"] or 0:.2e} | head LR={row["head_lr"] or 0:.2e}',flush=True)
        if stale>=args.patience:
            print('Early stopping.');break
    best_checkpoint=finalize_checkpoint(args.output_dir/'best.pt',pairs,args,device,model)
    finalize_checkpoint(args.output_dir/'last.pt',pairs,args,device,model)
    write_json(args.output_dir/'threshold_selection.json',best_checkpoint['threshold_selection'])
    write_json(args.output_dir/'calibration.json',best_checkpoint['calibration'])
def save_sanity(pairs: pd.DataFrame, args) -> None:
    sample = pd.concat([frame.sample(min(2, len(frame)), random_state=args.seed)
                        for _, frame in pairs.groupby('label')])
    remaining = pairs.drop(index=sample.index)
    if len(sample) < min(4, len(pairs)):
        sample = pd.concat([sample, remaining.sample(min(4 - len(sample), len(remaining)), random_state=args.seed)])
    dataset = PairedINbreastDataset(sample, args.size, False)
    fig, axes = plt.subplots(4, 2, figsize=(10, 18), squeeze=False)
    for row_index in range(4):
        for column, view in enumerate(("cc", "mlo")):
            axis = axes[row_index, column]
            if row_index >= len(dataset): axis.axis("off"); continue
            item = dataset[row_index]; image = denormalize(item[f"{view}_image"]); mask = item[f"{view}_mask"][0].numpy()
            axis.imshow(image, cmap="gray"); axis.imshow(np.ma.masked_where(mask == 0, mask), cmap="Reds", alpha=.45)
            axis.set_title(f"{item['pair_id']} {view.upper()} label={int(item['label'])}"); axis.axis("off")
    path = args.output_dir / "sanity" / "paired_overlays.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); print(f"Sanity check: {path}")


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    return np.clip(tensor[0].detach().cpu().numpy() * .229 + .485, 0, 1)


def save_history(history: list[dict], output: Path) -> None:
    frame = pd.DataFrame(history); frame.to_csv(output / "history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); axes[0].plot(frame.epoch, frame.train_loss, label="train"); axes[0].plot(frame.epoch, frame.val_loss, label="validation")
    axes[0].legend(); axes[0].set_title("Loss"); axes[1].plot(frame.epoch, frame.f1_at_05); axes[1].set_title("Validation F1 @ 0.5")
    fig.tight_layout(); fig.savefig(output / "training_curves.png", dpi=150); plt.close(fig)




def collect_localization(model,frame,args,device,save=False,heatmap_threshold=.5,classification_threshold=.5,calibration=None):
    if not getattr(args,'gradcam_enabled',False) or getattr(args,'gradcam_examples',8)==0:
        return [],[],[]
    decision_rows=predict_frame(model,frame,args,device)
    candidates=[]
    for row in decision_rows:
        p=float(apply_calibration([row['probability']],calibration,
                                 logits=[row['raw_logit']] if 'raw_logit' in row else None)[0])
        truth=row['true_label'];prediction=int(p>=classification_threshold)
        category=('TP' if prediction else 'FN') if truth==1 else (('FP' if prediction else 'TN') if truth==0 else 'unlabelled')
        if truth is None or category in getattr(args,'gradcam_categories',['FN','FP','TP','TN']):
            candidates.append((row['pair_id'],category,p))
    categories={pair_id:(category,p) for pair_id,category,p in candidates}
    priority = {'FN': 0, 'FP': 1, 'TP': 2, 'TN': 3, 'unlabelled': 4}
    ranked = sorted(candidates, key=lambda item: (priority[item[1]], -abs(item[2] - .5), item[0]))
    chosen=[item[0] for item in ranked[:getattr(args,'gradcam_examples',8)]]
    if not chosen:
        return [],[],[]
    selected=frame[frame.pair_id.isin(chosen)].sort_values('pair_id')
    loader=make_loader(selected,args,False,batch_size=1)
    heatmaps,masks,records=[],[],[]
    for batch in loader:
        cc,mlo=batch['cc_image'].to(device),batch['mlo_image'].to(device)
        cc_cam,mlo_cam=paired_gradcam(model,cc,mlo,target_class=1)
        truth=float(batch['label'][0]);true_label=int(truth) if np.isfinite(truth) else None
        pair_id=batch['pair_id'][0]
        category,probability=categories[pair_id]
        for view,cam in (('CC',cc_cam),('MLO',mlo_cam)):
            prefix=view.lower()
            geometry={key:int(value[0]) for key,value in batch[f'{prefix}_geometry'].items()}
            valid=geometry_valid_region(geometry)
            model_heatmap=cam.copy()
            total_heat=float(model_heatmap.sum())
            padding_heat_fraction=float(model_heatmap[~valid].sum()/total_heat) if total_heat>0 else 0.
            cam=np.where(valid,cam,0.)
            original_cam=heatmap_to_original(cam,geometry)
            xml=batch[f'{prefix}_xml_path'][0]
            original_shape=(geometry['original_height'],geometry['original_width'])
            roi=load_xml_mask(Path(xml),original_shape) if xml else np.zeros(original_shape,np.uint8)
            has_roi=bool(batch[f'{prefix}_has_roi'][0])
            original_valid=np.zeros(original_shape,bool)
            original_valid[geometry['crop_top']:geometry['crop_bottom'],geometry['crop_left']:geometry['crop_right']]=True
            record={'pair_id':pair_id,'patient_id':batch['patient_id'][0],'acquisition_date':batch['acquisition_date'][0],
                    'side':batch['side'][0],'view':view,'true_label':true_label,'probability':probability,
                    'predicted_label':int(probability>=classification_threshold),'category':category,
                    'classification_threshold':classification_threshold,'has_roi':has_roi,
                    'cam':original_cam,'mask':roi,'valid_region':original_valid,'geometry':geometry,
                    'target_class':1,'target_layer':model.gradcam_target_name,
                    'evaluation_space':'original DICOM resolution; square padding excluded'}
            record['padding_heat_fraction']=padding_heat_fraction
            if has_roi and roi.any():
                heatmaps.append(original_cam);masks.append(roi)
            if save:
                name=f'{pair_id}_{view}'.replace('/','_').replace('\\','_')
                title=f'{pair_id} {view} | true={true_label} p={probability:.6g} threshold={classification_threshold:.6g} pred={record["predicted_label"]}'
                destination=getattr(args,'gradcam_dir',None) or args.output_dir
                original_image=read_dicom(batch[f'{prefix}_dicom_path'][0])
                record.update(save_prediction_artifacts(original_image,original_cam,roi,heatmap_threshold,destination,name,title,
                              model_input=denormalize(batch[f'{prefix}_image'][0]),geometry=geometry,valid_region=original_valid,
                              model_heatmap=model_heatmap))
            records.append(record)
    return heatmaps,masks,records


def localization_rows(records,heatmap_threshold,fold=None):
    all_rows,metric_rows=[],[]
    for record in records:
        eligible=bool(record['has_roi'] and record['mask'].any())
        region=(record['cam']>=heatmap_threshold)&record.get('valid_region',np.ones(record['cam'].shape,bool))
        dice,iou=dice_iou(region,record['mask']) if eligible else (None,None)
        row={key:value for key,value in record.items() if key not in {'cam','mask','valid_region'}}
        row.update(heatmap_threshold=float(heatmap_threshold),eligible_for_localization_metrics=eligible,dice=dice,iou=iou,
                   skip_reason=None if eligible else ('no_roi' if not record['has_roi'] else 'empty_roi'))
        if fold is not None:
            row['fold']=fold
        all_rows.append(row)
        if eligible:
            metric_rows.append(row)
    return all_rows,metric_rows


def localization_coverage(frame,records,args):
    roi_counts={view:int(pd.to_numeric(frame.get(f'{view.lower()}_has_roi',pd.Series(0,index=frame.index))).sum()) for view in ('CC','MLO')}
    evaluated=sum(bool(row['has_roi'] and row['mask'].any()) for row in records)
    reasons={}
    missing=2*len(frame)-len(records)
    if missing:
        reasons['disabled' if not getattr(args,'gradcam_enabled',False) else 'example_limit_or_category_filter']=missing
    for record in records:
        if not record['has_roi'] or not record['mask'].any():
            reason='no_roi' if not record['has_roi'] else 'empty_roi'
            reasons[reason]=reasons.get(reason,0)+1
    return {'cc_images_with_roi':roi_counts['CC'],'mlo_images_with_roi':roi_counts['MLO'],
            'generated_heatmaps':len(records),'evaluated_heatmaps':evaluated,'skipped_heatmaps':2*len(frame)-evaluated,
            'skip_reasons':reasons,'threshold_source':'fixed configuration; never optimized on training or test images',
            'limitation':'Reported metrics cover only the configured subset; they are not whole-dataset localization estimates.'}


def assert_held_out(frame,checkpoint):
    used=set()
    for role,ids in checkpoint.get('patient_roles',{}).items():
        if role!='test':
            used.update(ids)
    overlap=used & set(frame.patient_id.astype(str))
    if overlap:
        raise ValueError(f'Finalna evaluacija ima patient leakage sa training/selection/calibration: {sorted(overlap)}')
    if not checkpoint.get('postprocessing_finalized',False):
        raise ValueError('Checkpoint nema završen izbor praga/kalibracije; nije validan za finalnu evaluaciju.')


def evaluate_model(pairs,args,device):
    model,checkpoint=load_checkpoint(args,device)
    role=getattr(args,'eval_split','test')
    evaluation=pairs[pairs.split==role]
    if evaluation.empty:
        raise ValueError(f'Evaluacioni split {role} je prazan; pripremite grouped holdout metadata.')
    if role=='test':
        assert_held_out(evaluation,checkpoint)
    rows=apply_decisions(predict_frame(model,evaluation,args,device),checkpoint)
    labels=[r['true_label'] for r in rows];p=[r['probability'] for r in rows]
    result={'evaluation_role':role,'final_estimate':role=='test','unit':'one breast CC/MLO pair',
        'classification_at_05':classification_metrics(labels,p,.5,args.calibration_bins),
        'classification_tuned':classification_metrics_from_predictions(labels,[r['prediction'] for r in rows],p,args.calibration_bins),
        'threshold_selection':checkpoint.get('threshold_selection'),'calibration':checkpoint.get('calibration'),
        'bootstrap':patient_cluster_bootstrap(labels,p,[r['patient_id'] for r in rows],predictions=[r['prediction'] for r in rows],
                    iterations=args.bootstrap_iterations,seed=args.seed,bins=args.calibration_bins)}
    _,_,records=collect_localization(model,evaluation,args,device,save=True,heatmap_threshold=args.heatmap_threshold,
                                    classification_threshold=checkpoint['threshold'],calibration=checkpoint.get('calibration'))
    localization,metric_rows=localization_rows(records,args.heatmap_threshold)
    result.update(localization=localization_summary(metric_rows),localization_coverage=localization_coverage(evaluation,records,args))
    prediction_artifacts(rows,args.output_dir,args.calibration_bins,pairs)
    pd.DataFrame(localization).to_csv(args.output_dir/'localization_predictions.csv',index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir/'localization_metrics.csv',index=False)
    write_json(args.output_dir/'metrics.json',result)
    if role!='test':
        result['warning']='Development diagnostic; these results must not be presented as final performance.'
        write_json(args.output_dir/'metrics.json',result)
    print(json.dumps(result,indent=2))


def predict_cases(frame,args,device):
    model,checkpoint=load_checkpoint(args,device)
    rows=apply_decisions(predict_frame(model,frame,args,device),checkpoint)
    _,_,records=collect_localization(model,frame,args,device,save=True,heatmap_threshold=args.heatmap_threshold,
                                    classification_threshold=checkpoint['threshold'],calibration=checkpoint.get('calibration'))
    localization,_=localization_rows(records,args.heatmap_threshold)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'single_prediction.csv',index=False)
    pd.DataFrame(localization).to_csv(args.output_dir/'single_localization.csv',index=False)
    write_json(args.output_dir/'single_prediction.json',rows)
    print(json.dumps(rows,indent=2,default=str))


def direct_prediction_frame(args):
    import pydicom
    identities=[]
    header_dates=[]
    date_sources=[]
    tables={}
    for path,expected in ((args.cc_path,'CC'),(args.mlo_path,'MLO')):
        if not path.is_file():
            raise FileNotFoundError(f'DICOM putanja ne postoji: {path}')
        ds=pydicom.dcmread(path,stop_before_pixels=True)
        name_side,name_view=_view_from_name(path)
        side=str(getattr(ds,'ImageLaterality','') or getattr(ds,'Laterality','') or name_side or '').upper()
        view=str(getattr(ds,'ViewPosition','') or name_view or '').upper()
        view='MLO' if view=='ML' else view
        patient=str(getattr(ds,'PatientID','')).strip() or _patient_from_name(path)
        date=str(getattr(ds,'AcquisitionDate','')).strip()
        header_dates.append(date)
        name_patient=_patient_from_name(path) if name_view else ''
        if name_patient and patient!=name_patient:
            raise ValueError(f'DICOM header/filename patient mismatch: {path}')
        # Reading a supplied dataset table is optional; inference never generates metadata.
        root=path.parent.parent if path.parent.name.lower()=='alldicoms' else path.parent
        if root not in tables:
            candidates=sorted(p for p in root.glob('*') if p.suffix.lower() in {'.xls','.xlsx','.csv'} and p.name.lower().startswith('inbreast'))
            mapping={}
            if candidates:
                table=_read_metadata(next((p for p in candidates if p.suffix.lower()=='.xls'),candidates[0]))
                file_col,date_col=_column(table,'filename','file'),_column(table,'acquisitiondate')
                mapping={image_key(row[file_col]):acquisition_key(row[date_col]) for _,row in table.iterrows() if not pd.isna(row[file_col])}
            tables[root]=mapping
        table_date=tables[root].get(image_key(path),'')
        if date and table_date and not (date.startswith(table_date) or table_date.startswith(date)):
            raise ValueError(f'DICOM acquisition_date/header/metadata mismatch: {path}')
        date_sources.append('dataset_table' if table_date else ('header' if date else 'unknown'))
        date=table_date or date
        if side not in {'L','R'} or not patient or view!=expected:
            raise ValueError(f'Direktan inference zahteva proverljiv patient/side/{expected} identitet: {path}')
        if name_side and name_side!=side or name_view and name_view!=view:
            raise ValueError(f'DICOM header/filename mismatch: {path}')
        identities.append((patient,date,side))
    if identities[0]!=identities[1]:
        raise ValueError('Direktan CC/MLO par mora imati istog pacijenta, acquisition date i stranu.')
    if all(header_dates) and header_dates[0]!=header_dates[1]:
        raise ValueError('Direktan CC/MLO par ima različite DICOM acquisition date vrednosti.')
    patient,date,side=identities[0]
    if not date:
        warnings.warn('Acquisition date ne postoji u oba DICOM headera niti u dostupnoj dataset tabeli; '
                      'datum direktnog para nije moguće proveriti. Koristite --pair-id kada postoji pripremljen pregled.',RuntimeWarning)
    for xml in (args.cc_xml,args.mlo_xml):
        if xml and not xml.is_file():
            raise FileNotFoundError(f'XML putanja ne postoji: {xml}')
    pair_id=args.pair_id or (f'{patient}_{date}_{side}' if date else f'{args.cc_path.stem}_{args.mlo_path.stem}')
    return pd.DataFrame([{'pair_id':pair_id,'patient_id':patient,'acquisition_date':date,'acquisition_date_source':'|'.join(date_sources),
        'side':side,'label':np.nan,'split':'predict',
        'cc_image_id':args.cc_path.stem.split('_')[0],'mlo_image_id':args.mlo_path.stem.split('_')[0],
        'cc_dicom_path':str(args.cc_path.resolve()),'cc_xml_path':str(args.cc_xml.resolve()) if args.cc_xml else '',
        'cc_has_roi':int(args.cc_xml is not None),'mlo_dicom_path':str(args.mlo_path.resolve()),
        'mlo_xml_path':str(args.mlo_xml.resolve()) if args.mlo_xml else '', 'mlo_has_roi':int(args.mlo_xml is not None)}])


def run_cross_validation(pairs,args,device):
    """Outer grouped stratified CV with independent development holdouts (not full nested hyperparameter CV)."""
    root=args.output_dir/'crossval';root.mkdir(parents=True,exist_ok=True)
    folded=assign_patient_folds(pairs,args.folds,args.seed)
    folded.to_csv(root/'metadata_pairs_cv.csv',index=False)
    prepared=[]
    for fold in range(1,args.folds+1):
        outer=folded[folded.fold==fold].copy()
        development=grouped_development_split(folded[folded.fold!=fold].drop(columns='fold'),args.seed+fold,
            args.inner_val_fraction,getattr(args,'threshold_fraction',.15),
            getattr(args,'calibration_fraction',.10) if getattr(args,'calibration_method','none')!='none' else 0.,0.)
        audit=split_audit(development,outer)
        prepared.append((fold,outer,development,audit))
    write_json(root/'patient_leakage_audit.json',{'patient_leakage':False,'outer_patient_fold_unique':True,
                                              'folds':{str(fold):audit for fold,_,_,audit in prepared}})
    all_rows,all_localization,all_metric_rows,fold_summaries=[],[],[],[]
    for fold,outer,development,audit in prepared:
        fold_args=copy.copy(args)
        fold_args.output_dir=root/f'fold_{fold}'
        fold_args.checkpoint=None;fold_args.resume=None;fold_args.seed=args.seed+fold
        fold_args.output_dir.mkdir(parents=True,exist_ok=True)
        development.to_csv(fold_args.output_dir/'metadata_pairs_development.csv',index=False)
        outer.assign(split='test').to_csv(fold_args.output_dir/'metadata_pairs_outer_test.csv',index=False)
        seed_everything(fold_args.seed)
        write_json(fold_args.output_dir/'split_audit.json',audit)
        print(f'CV fold {fold}/{args.folds}; all patient intersections empty',flush=True)
        train_model(development,fold_args,device)
        model,checkpoint=load_checkpoint(fold_args,device)
        if 'patient_roles' in checkpoint:
            assert_held_out(outer,checkpoint)
        rows=apply_decisions(predict_frame(model,outer,fold_args,device),checkpoint,fold)
        labels=[r['true_label'] for r in rows];probabilities=[r['probability'] for r in rows]
        fixed=classification_metrics(labels,probabilities,.5,getattr(args,'calibration_bins',10))
        tuned=classification_metrics_from_predictions(labels,[r['prediction'] for r in rows],probabilities,getattr(args,'calibration_bins',10))
        _,_,records=collect_localization(model,outer,fold_args,device,save=True,
                    heatmap_threshold=getattr(args,'heatmap_threshold',.5),classification_threshold=checkpoint['threshold'],
                    calibration=checkpoint.get('calibration'))
        localization,metric_rows=localization_rows(records,getattr(args,'heatmap_threshold',.5),fold)
        local=localization_summary(metric_rows)
        result={'fold':fold,'unit':'one breast CC/MLO pair','outer_patients':int(outer.patient_id.nunique()),'outer_pairs':len(outer),
            'classification_at_05':fixed,'classification_inner_tuned':tuned,'threshold_selection':checkpoint.get('threshold_selection'),
            'calibration':checkpoint.get('calibration'),'localization':local,
            'localization_coverage':localization_coverage(outer,records,args),
            'bootstrap':patient_cluster_bootstrap(labels,probabilities,[r['patient_id'] for r in rows],
                predictions=[r['prediction'] for r in rows],iterations=getattr(args,'bootstrap_iterations',2000),seed=fold_args.seed,
                bins=getattr(args,'calibration_bins',10))}
        write_json(fold_args.output_dir/'metrics.json',result)
        prediction_artifacts(rows,fold_args.output_dir,getattr(args,'calibration_bins',10),pairs)
        pd.DataFrame(localization).to_csv(fold_args.output_dir/'localization_predictions.csv',index=False)
        pd.DataFrame(metric_rows).to_csv(fold_args.output_dir/'localization_metrics.csv',index=False)
        summary={'fold':fold,'outer_patients':int(outer.patient_id.nunique()),'outer_pairs':len(outer),
                 'f1_at_05':fixed['f1'],'f1_inner_tuned':tuned['f1'],'threshold':checkpoint['threshold'],
                 'dice_mean':local['all']['dice_mean'],'iou_mean':local['all']['iou_mean']}
        summary.update({name:tuned[name] for name in METRIC_NAMES})
        fold_summaries.append(summary)
        all_rows.extend(rows);all_localization.extend(localization);all_metric_rows.extend(metric_rows)
        print(f'Fold {fold}: PR-AUC={tuned["pr_auc"]}, sensitivity={tuned["sensitivity"]}, specificity={tuned["specificity"]}',flush=True)
        del model
        if device.type=='cuda':
            torch.cuda.empty_cache()
    frame=pd.DataFrame(all_rows)
    if len(frame)!=len(pairs) or frame.pair_id.duplicated().any() or set(frame.pair_id)!=set(pairs.pair_id):
        raise RuntimeError('OOF nema tačno jednu predikciju za svaki par.')
    if frame.groupby('patient_id').fold.nunique().max()!=1:
        raise RuntimeError('OOF patient leakage između spoljašnjih foldova.')
    fold_frame=pd.DataFrame(fold_summaries)
    mean_std={}
    for column in [*METRIC_NAMES,'f1_at_05','f1_inner_tuned','dice_mean','iou_mean']:
        values=pd.to_numeric(fold_frame[column],errors='coerce').dropna()
        mean_std[column]={'mean':float(values.mean()) if len(values) else None,
                          'std':float(values.std(ddof=1)) if len(values)>1 else None,'defined_folds':len(values)}
    result={'folds':args.folds,'patients':int(pairs.patient_id.nunique()),'pairs':len(pairs),'unit':'one breast CC/MLO pair',
        'classification_at_05':classification_metrics(frame.true_label,frame.probability,.5,getattr(args,'calibration_bins',10)),
        'classification_inner_tuned_per_fold':classification_metrics_from_predictions(frame.true_label,frame.prediction,frame.probability,getattr(args,'calibration_bins',10)),
        'bootstrap':patient_cluster_bootstrap(frame.true_label,frame.probability,frame.patient_id,predictions=frame.prediction,
                                              iterations=getattr(args,'bootstrap_iterations',2000),seed=args.seed,bins=getattr(args,'calibration_bins',10)),
        'fold_mean_std':mean_std,'localization':localization_summary(all_metric_rows),
        'patient_leakage':False,'method':'Outer stratified patient-grouped CV; disjoint train/checkpoint/threshold/optional calibration holdouts within each development fold; no hyperparameter search',
        'localization_policy':'Fixed Grad-CAM threshold; only nonempty XML ROI views; original DICOM space; padding excluded; configured subset only'}
    prediction_artifacts(all_rows,root,getattr(args,'calibration_bins',10),pairs)
    frame.sort_values(['fold','pair_id']).to_csv(root/'oof_predictions.csv',index=False)
    fold_frame.to_csv(root/'fold_metrics.csv',index=False)
    pd.DataFrame(all_localization).to_csv(root/'localization_predictions.csv',index=False)
    pd.DataFrame(all_metric_rows).to_csv(root/'localization_metrics.csv',index=False)
    write_json(root/'metrics.json',result)
    print(json.dumps(result,indent=2))


def choose_device(name='auto'):
    if name=='auto':
        name='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name=='cuda' and not torch.cuda.is_available() or name=='mps' and not torch.backends.mps.is_available():
        raise ValueError(f'Uređaj {name} nije dostupan.')
    return torch.device(name)


def main(argv=None):
    args=parse_args(argv);seed_everything(args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.mode=='predict' and not args.pair_id and not (args.cc_path or args.mlo_path):
        raise ValueError('Predict zahteva --pair-id ili --cc-path i --mlo-path.')
    metadata=args.metadata_pairs or args.output_dir/'metadata_pairs.csv'
    if args.mode=='prepare' or args.rebuild_metadata:
        if args.data_root is None:
            raise ValueError('--data-root je obavezan za prepare/--rebuild-metadata.')
        _,pairs=build_metadata(args.data_root,args.output_dir,args.metadata_path,args.seed,split_config=args)
        if args.mode=='prepare':
            write_json(args.output_dir/'split_audit.json',split_audit(pairs))
            if args.metadata_pairs and args.metadata_pairs.resolve()!=(args.output_dir/'metadata_pairs.csv').resolve():
                args.metadata_pairs.parent.mkdir(parents=True,exist_ok=True)
                pairs.to_csv(args.metadata_pairs,index=False)
            return
    elif args.mode=='predict' and (args.cc_path or args.mlo_path):
        if not args.cc_path or not args.mlo_path:
            raise ValueError('Direktna predikcija zahteva oba CC/MLO ulaza.')
        pairs=direct_prediction_frame(args)
    else:
        pairs=load_metadata_pairs(metadata,check_files=True)
    if args.mode=='predict' and args.pair_id and not (args.cc_path or args.mlo_path):
        pairs=pairs[pairs.pair_id==args.pair_id].copy()
        if pairs.empty:
            raise ValueError(f'Pair ID nije pronađen: {args.pair_id}')
    if args.mode=='sanity':
        write_json(args.output_dir/'split_audit.json',split_audit(pairs))
        save_sanity(pairs,args);return
    device=choose_device(args.device)
    print(f'Device: {device}; pairs={len(pairs)}, patients={pairs.patient_id.nunique()}',flush=True)
    if args.mode=='train':
        train_model(pairs,args,device)
    elif args.mode=='crossval':
        run_cross_validation(pairs,args,device)
    elif args.mode=='predict':
        predict_cases(pairs,args,device)
    else:
        evaluate_model(pairs,args,device)


if __name__=='__main__':
    main()
