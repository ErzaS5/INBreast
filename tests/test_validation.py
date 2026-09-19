from pathlib import Path
import json
import plistlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest
import torch
from torch import nn

import data
import evaluate
import train
from configuration import parse_args
from data import (assign_patient_folds, build_metadata, grouped_development_split, load_metadata_pairs,
                  patient_train_val_split, select_pairing_views, split_audit, validate_metadata_pairs)
from evaluate import (apply_calibration, calibration_metrics, classification_metrics, classification_metrics_from_predictions,
                      fit_temperature, patient_cluster_bootstrap, select_threshold)
from gradcam import _to_cam, normalize_cam_in_valid_region, paired_gradcam
from model import ARCHITECTURE_NAME, ARCHITECTURE_VERSION, NORMALIZATION, create_model
from preprocessing import (geometry_valid_region, heatmap_to_original, load_xml_mask, normalize_dicom_pixels, prepare)


def write_dicom(path, patient='p0', date='20200101', side='L', view='CC', offset=0):
    meta=FileMetaDataset()
    meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=pydicom.uid.DigitalMammographyXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=meta.MediaStorageSOPClassUID;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.PatientID=patient;ds.AcquisitionDate=date;ds.ImageLaterality=side;ds.ViewPosition=view
    ds.Rows=80;ds.Columns=50;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=16;ds.BitsStored=12;ds.HighBit=11;ds.PixelRepresentation=0
    image=np.zeros((80,50),np.uint16)
    image[5:75,4:46]=np.arange(70*42,dtype=np.uint16).reshape(70,42)+offset+10
    ds.PixelData=image.tobytes();ds.save_as(path,enforce_file_format=True)
    return path


@pytest.fixture
def pair(tmp_path):
    cc=write_dicom(tmp_path/'1_p0_MG_L_CC_ANON.dcm')
    mlo=write_dicom(tmp_path/'2_p0_MG_L_MLO_ANON.dcm',view='MLO')
    return pd.DataFrame([{'pair_id':'p0_20200101_L','patient_id':'p0','acquisition_date':'20200101','exam_id':'p0_20200101',
        'side':'L','label':1,'split':'train','metadata_schema_version':2,'preprocessing_version':2,
        'cc_dicom_path':str(cc),'mlo_dicom_path':str(mlo),'cc_xml_path':'','mlo_xml_path':'','cc_has_roi':0,'mlo_has_roi':0}])


@pytest.mark.parametrize('empty', ['blank','header'])
def test_empty_metadata(tmp_path,empty):
    path=tmp_path/'pairs.csv'
    path.write_text('' if empty=='blank' else 'pair_id,patient_id\n')
    with pytest.raises(ValueError):
        load_metadata_pairs(path)


@pytest.mark.parametrize('column',['patient_id','label','cc_dicom_path','mlo_dicom_path','acquisition_date','metadata_schema_version'])
def test_missing_metadata_columns(pair,column):
    with pytest.raises(ValueError,match='obavezne kolone'):
        validate_metadata_pairs(pair.drop(columns=column))


def test_duplicate_pair_id(pair):
    with pytest.raises(ValueError,match='duplirane pair_id'):
        validate_metadata_pairs(pd.concat([pair,pair]))


@pytest.mark.parametrize('prefix',['cc','mlo'])
def test_missing_dicom_path(pair,prefix):
    pair.loc[0,f'{prefix}_dicom_path']='absent.dcm'
    with pytest.raises(FileNotFoundError,match='DICOM'):
        validate_metadata_pairs(pair)


def test_incomplete_pair(pair):
    pair.loc[0,'mlo_dicom_path']=''
    with pytest.raises(ValueError,match='nepotpun'):
        validate_metadata_pairs(pair)


@pytest.mark.parametrize('field,value',[('ImageLaterality','R'),('AcquisitionDate','20200201'),('PatientID','p1'),('ViewPosition','CC')])
def test_dicom_pair_identity_mismatch(pair,field,value):
    path=Path(pair.loc[0,'mlo_dicom_path'])
    ds=pydicom.dcmread(path);setattr(ds,field,value);ds.save_as(path,enforce_file_format=True)
    with pytest.raises(ValueError,match='mismatch'):
        validate_metadata_pairs(pair)


@pytest.mark.parametrize('column,value',[('cc_side','R'),('mlo_acquisition_date','20200201'),('mlo_patient_id','p1'),('cc_view','MLO')])
def test_declared_identity_mismatch(pair,column,value):
    pair[column]=value
    with pytest.raises(ValueError,match='mismatch'):
        validate_metadata_pairs(pair)


@pytest.mark.parametrize('column,value',[('label',2),('label',np.nan),('side','U'),('metadata_schema_version',2.1),('metadata_schema_version',np.nan),('preprocessing_version',1)])
def test_invalid_metadata_values(pair,column,value):
    pair[column] = pair[column].astype(object)
    pair.loc[0,column]=value
    with pytest.raises(ValueError):
        validate_metadata_pairs(pair)


def grouped_frame(n=40):
    return pd.DataFrame([{'patient_id':f'p{i}','pair_id':f'p{i}_{date}_{side}','acquisition_date':date,
                         'side':side,'label':i%2,'split':'train'}
                        for i in range(n) for date in ('202001','202101') for side in ('L','R')])


def test_multiple_dates_both_breasts_grouped_and_reproducible():
    frame=grouped_frame()
    first=grouped_development_split(frame,seed=4,val_fraction=.15,threshold_fraction=.15,calibration_fraction=.1,test_fraction=.2)
    second=grouped_development_split(frame,seed=4,val_fraction=.15,threshold_fraction=.15,calibration_fraction=.1,test_fraction=.2)
    pd.testing.assert_frame_equal(first,second)
    assert first.groupby('patient_id').split.nunique().eq(1).all()
    assert set(first.split)=={'train','val','threshold','calibration','test'}
    assert all(not ids for ids in split_audit(first)['intersections'].values())
    folds=assign_patient_folds(frame,5,4)
    assert folds.groupby('patient_id').fold.nunique().eq(1).all()
    pd.testing.assert_frame_equal(folds,assign_patient_folds(frame,5,4))


def test_patient_leakage_detection():
    frame=grouped_frame();frame.loc[0,'split']='test'
    with pytest.raises(ValueError,match='leakage'):
        split_audit(frame)


@pytest.mark.parametrize('label',[0,1])
def test_split_requires_both_classes(label):
    frame=grouped_frame();frame['label']=label
    with pytest.raises(ValueError,match='Stratifikovana'):
        patient_train_val_split(frame)
    with pytest.raises(ValueError,match='obe klase'):
        assign_patient_folds(frame)


def test_too_small_stratified_split_fails_without_random_fallback():
    with pytest.raises(ValueError,match='random fallback nije dozvoljen'):
        patient_train_val_split(grouped_frame(4),val_fraction=.25)


def test_repeated_exposures_rule_and_order_invariance():
    frame=pd.DataFrame([{'patient_id':'p','acquisition_date':'202001','side':'L','view':'CC','image_id':str(i),
                         'label':label,'has_roi':roi}
                        for i,label,roi in [(1,0,1),(2,1,0),(3,1,1),(4,1,1)]])
    selected,audit=select_pairing_views(frame)
    assert selected.image_id.tolist()==['3']
    assert len(audit)==4 and audit.selected_for_pairing.sum()==1
    selected2,_=select_pairing_views(frame.sample(frac=1,random_state=8))
    assert selected2.image_id.tolist()==['3']


@pytest.fixture
def synthetic_root(tmp_path):
    root=tmp_path/'dataset';root.mkdir()
    table=[]
    for i in range(20):
        for view_index,view in enumerate(('CC','MLO')):
            image_id=str(100+i*2+view_index)
            write_dicom(root/f'{image_id}_p{i}_MG_L_{view}_ANON.dcm',patient=f'p{i}',view=view,offset=i)
            with (root / f'{image_id}.xml').open('wb') as stream:
                plistlib.dump({'Images': [{'ROIs': [{'Point_px': ['(15, 20)', '(30, 20)', '(30, 35)', '(15, 35)']}]}]}, stream)
            table.append({'File Name':int(image_id),'Bi-Rads':1 if i%2==0 else 4,'Acquisition date':20200101,
                          'Laterality':'L','View':view})
    pd.DataFrame(table).to_csv(root/'INbreast.csv',index=False)
    return root


def test_prepare_single_complete_breast_and_conflicting_birads(synthetic_root,tmp_path):
    table=pd.read_csv(synthetic_root/'INbreast.csv');table.loc[0,'Bi-Rads']=3;table.loc[1,'Bi-Rads']=4
    table.to_csv(synthetic_root/'INbreast.csv',index=False)
    args=parse_args(['--size','64'])
    _,pairs=build_metadata(synthetic_root,tmp_path/'prepared',split_config=args)
    p0=pairs[pairs.patient_id=='p0'].iloc[0]
    assert p0.label==1 and p0.label_conflict==1 and p0.birads_conflict==1
    assert len(pairs)==20 and set(pairs.side)=={'L'}
    assert (tmp_path/'prepared'/'incomplete_pairs.csv').is_file()
    validate_metadata_pairs(pairs)


def test_prepare_records_incomplete_pairs(synthetic_root,tmp_path):
    (synthetic_root/'101_p0_MG_L_MLO_ANON.dcm').unlink()
    _,pairs=build_metadata(synthetic_root,tmp_path/'prepared',split_config=parse_args(['--size','64']))
    audit=pd.read_csv(tmp_path/'prepared'/'incomplete_pairs.csv')
    assert len(audit)==1 and audit.iloc[0].reason=='missing_MLO'
    assert 'p0' not in set(pairs.patient_id)


def test_direct_unlabelled_inference_and_partial_input(pair):
    args=parse_args(['--cc-path',pair.cc_dicom_path[0],'--mlo-path',pair.mlo_dicom_path[0],'--size','64'])
    frame=train.direct_prediction_frame(args)
    assert np.isnan(frame.iloc[0].label) and frame.iloc[0].patient_id=='p0'
    assert not frame.iloc[0].cc_xml_path
    item=data.PairedINbreastDataset(frame,size=64)[0]
    assert torch.isnan(item['label'])
    assert item['cc_image'].shape==(3,64,64)


@pytest.mark.parametrize('probabilities',[[np.nan,.3],[np.inf,.3],[-.1,.3],[1.1,.3],[.3]])
def test_invalid_probabilities(probabilities):
    with pytest.raises(ValueError,match='Verovatnoće'):
        classification_metrics([0,1],probabilities)


@pytest.mark.parametrize('label',[0,1])
def test_single_class_evaluation_has_stable_null_metrics(label):
    result=classification_metrics([label]*3,[.1,.5,.9])
    assert result['roc_auc'] is None and result['pr_auc'] is None
    assert result['warnings'] and len(result['confusion_matrix'])==2
    assert all(name in result for name in evaluate.METRIC_NAMES)
    json.dumps(result,allow_nan=False)


def test_all_required_metrics_perfect_predictions():
    result=classification_metrics([0,0,1,1],[.1,.2,.8,.9])
    for name in ['accuracy','balanced_accuracy','sensitivity','specificity','precision','npv','f1','mcc','roc_auc','pr_auc']:
        assert result[name]==1.
    assert result['brier_score']==pytest.approx(.025)


@pytest.mark.parametrize('strategy',['fixed','max_f1','youden_j','min_sensitivity'])
def test_threshold_strategies(strategy):
    result=select_threshold([0,0,1,1],[.11,.7,.6,.9],strategy=strategy,minimum_sensitivity=1.,source='independent_test_fixture')
    assert 0<=result['threshold']<=1 and result['source']=='independent_test_fixture'
    if strategy=='min_sensitivity':
        assert result['threshold']==.6 and result['achieved_sensitivity']==1.
        assert result['achieved_specificity']==.5


def test_threshold_uses_exact_probabilities_and_deterministic_ties():
    y=[0,1,1];p=[.002,.003123,.999]
    first=select_threshold(y,p,strategy='min_sensitivity',minimum_sensitivity=1.)
    assert first['threshold']==.003123
    assert first==select_threshold(y,p,strategy='min_sensitivity',minimum_sensitivity=1.)


def test_threshold_single_class_requires_fixed():
    with pytest.raises(ValueError,match='obe klase'):
        select_threshold([0,0],[.1,.2])
    assert select_threshold([0,0],[.1,.2],strategy='fixed')['threshold']==.5


def test_patient_bootstrap_repeats_whole_clusters_and_is_reproducible(monkeypatch):
    original=evaluate.classification_metrics_from_predictions
    sampled=[]
    def capture(labels,predictions,probabilities,bins=10):
        sampled.append(list(probabilities))
        return original(labels,predictions,probabilities,bins)
    monkeypatch.setattr(evaluate,'classification_metrics_from_predictions',capture)
    result=patient_cluster_bootstrap([0,1,0,1],[.1,.2,.8,.9],['p1','p1','p2','p2'],iterations=30,seed=8)
    assert all(values.count(.1)==values.count(.2) and values.count(.8)==values.count(.9) for values in sampled)
    assert result==patient_cluster_bootstrap([0,1,0,1],[.1,.2,.8,.9],['p1','p1','p2','p2'],iterations=30,seed=8)
    assert result['iterations']==30 and result['patients']==2


def test_temperature_calibration_improvement_and_declined_no_improvement():
    fitted=fit_temperature([0,1,0,1],[.1,.9,.2,.8])
    assert fitted['applied'] and fitted['after_candidate']['brier_score']<fitted['before']['brier_score']
    assert np.isfinite(apply_calibration([.1,.9],fitted)).all()
    unchanged=fit_temperature([0,1],[.5,.5])
    assert not unchanged['applied']
    assert np.array_equal(apply_calibration([.1,.9],unchanged),[.1,.9])


@pytest.mark.parametrize('values',[[np.nan,1.],[np.inf,1.]])
def test_nonfinite_pixels_are_reported_and_sanitized(values):
    with pytest.warns(RuntimeWarning,match='NaN/Inf'):
        normalized=normalize_dicom_pixels(np.array(values))
    assert np.isfinite(normalized).all() and normalized[0]==0


def test_all_nonfinite_pixels_fail():
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError,match='validne piksele'):
            normalize_dicom_pixels(np.full((2,2),np.nan))


def test_xml_polygon_outside_image_is_clipped(tmp_path):
    path=tmp_path/'roi.xml'
    with path.open('wb') as stream:
        plistlib.dump({'Images':[{'ROIs':[{'Point_px':['(-4, -4)','(20, -4)','(20, 20)','(-4, 20)']}]}]},stream)
    mask=load_xml_mask(path,(12,12))
    assert mask.shape==(12,12) and mask.sum()==144


def test_one_pixel_roi_survives_extreme_resize():
    image=np.full((2048,1024),.7,np.float32)
    mask=np.zeros(image.shape,np.uint8);mask[1000,500]=1
    _,resized=prepare(image,mask,32)
    assert resized.sum()>0


def test_geometry_roundtrip_padding_excluded():
    image=np.zeros((100,50),np.float32);image[10:90,5:45]=.7
    mask=np.zeros_like(image,np.uint8);mask[40:60,20:30]=1
    _,roi,geometry=prepare(image,mask,64,return_geometry=True)
    valid=geometry_valid_region(geometry)
    assert not valid.all() and valid[roi.astype(bool)].all()
    restored=heatmap_to_original(roi.astype(np.float32),geometry)
    assert restored.shape==image.shape and restored[50,25]>.5
    padding=np.where(valid,0.,1.).astype(np.float32)
    assert not heatmap_to_original(padding,geometry).any()


def test_metadata_reader_strict_checks_files_when_enabled(pair,tmp_path):
    pair.cc_dicom_path='missing.dcm';path=tmp_path/'pairs.csv';pair.to_csv(path,index=False)
    assert len(load_metadata_pairs(path,check_files=False))==1
    with pytest.raises(FileNotFoundError):
        load_metadata_pairs(path,check_files=True)


def valid_checkpoint():
    return {'preprocessing_version':2,'metadata_schema_version':2,'architecture_name':ARCHITECTURE_NAME,
            'architecture_version':ARCHITECTURE_VERSION,'normalization':NORMALIZATION,'num_outputs':1,
            'patient_roles':{'train':['p0'],'val':['p1'],'threshold':['p2']},
            'label_definition':train.LABEL_DEFINITION,'size':64,'dropout':.3,'backbone':'swin_tiny_patch4_window7_224','threshold':.5}


@pytest.mark.parametrize('field,value',[('preprocessing_version',1),('metadata_schema_version',1),('architecture_version',99),
                                      ('normalization',{}),('threshold',np.nan),('size',63),('label_definition','changed')])
def test_incompatible_checkpoint_rejected(field,value):
    checkpoint=valid_checkpoint();checkpoint[field]=value
    with pytest.raises(ValueError):
        train.validate_checkpoint_versions(checkpoint)


def test_final_test_cannot_overlap_any_selection_role():
    frame=pd.DataFrame({'patient_id':['p1']})
    for role in ('train','val','threshold','calibration'):
        with pytest.raises(ValueError,match='leakage'):
            train.assert_held_out(frame,{'patient_roles':{role:['p1']},'postprocessing_finalized':True})


@pytest.mark.parametrize('key,value',[('dropout',1.),('batch_size',0),('lr',0.),('epochs',-1),('folds',1),
                                     ('fixed_threshold',1.1),('minimum_sensitivity',-.1),('grouped_split',False),
                                     ('unknown',1),('workers',-1),('size',63),('batch_size',True)])
def test_invalid_yaml_schema(tmp_path,key,value):
    import yaml
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump({key:value}))
    with pytest.raises(ValueError):
        parse_args(['--config',str(path)])


def test_cli_yaml_precedence_and_types(tmp_path):
    path=tmp_path/'config.yaml'
    path.write_text('epochs: 7\nsize: 64\noutput_dir: custom\nthreshold:\n  strategy: youden_j\n  fixed_value: 0.3\n')
    args=parse_args(['--config',str(path),'--epochs','9','--fixed-threshold','0.4','--no-pretrained'])
    assert args.epochs==9 and args.size==64 and args.workers==0 and args.no_pretrained
    assert args.fixed_threshold==.4 and args.threshold_strategy=='youden_j'
    assert args.output_dir==Path('custom')


@pytest.mark.parametrize('backbone',['swin_tiny_patch4_window7_224','resnet18','densenet121'])
def test_supported_backbones_parse(backbone):
    assert parse_args(['--backbone',backbone]).backbone == backbone


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError,match='backbone'):
        parse_args(['--backbone','unknown_network'])


def test_unknown_nested_yaml_key(tmp_path):
    path=tmp_path/'config.yaml';path.write_text('threshold:\n  typo: 1\n')
    with pytest.raises(ValueError,match='Nepoznati'):
        parse_args(['--config',str(path)])


def test_gradcam_layout_finiteness_constant_and_shapes():
    activation=torch.ones(1,4,4,3);gradient=torch.ones_like(activation)
    cam=_to_cam(activation,gradient,(16,16),'NHWC')
    assert cam.shape==(16,16) and not cam.any()
    assert np.array_equal(cam,_to_cam(activation.permute(0,3,1,2),gradient.permute(0,3,1,2),(16,16),'NCHW'))
    with pytest.raises(ValueError,match='layout'):
        _to_cam(activation,gradient,(16,16),'unknown')
    with pytest.raises(ValueError,match='NaN/Inf'):
        _to_cam(activation,gradient*float('nan'),(16,16))


def test_gradcam_is_renormalized_after_padding_is_excluded():
    cam=np.array([[1.,.8,0.],[.4,.2,0.]],np.float32)
    valid=np.array([[False,True,False],[True,True,False]])
    normalized=normalize_cam_in_valid_region(cam,valid)
    assert normalized[~valid].sum()==0
    assert normalized[valid].min()==0
    assert normalized[valid].max()==1


@pytest.mark.parametrize('invalid',['shape','batch','finite','size'])
def test_model_input_validation(invalid):
    if invalid=='size':
        with pytest.raises(ValueError,match='deljiv'):
            create_model(size=63,pretrained=False)
        return
    model=create_model(size=64,pretrained=False)
    cc=torch.rand(1,3,64,64);mlo=cc.clone()
    if invalid=='shape':cc=cc[:,0]
    if invalid=='batch':mlo=mlo.repeat(2,1,1,1)
    if invalid=='finite':cc[0,0,0,0]=float('nan')
    with pytest.raises(ValueError):
        model(cc,mlo)


def test_loader_generator_reproducible_and_state_restorable(pair):
    frame=pd.concat([pair.assign(pair_id=f'pair{i}') for i in range(8)],ignore_index=True)
    args=parse_args(['--size','64','--batch-size','2','--seed','3','--no-augmentations'])
    first=train.make_loader(frame,args,True);second=train.make_loader(frame,args,True)
    assert [b['pair_id'] for b in first]==[b['pair_id'] for b in second]
    state=first.generator.get_state()
    expected=[b['pair_id'] for b in first]
    second.generator.set_state(state)
    assert expected==[b['pair_id'] for b in second]


class TinyResumeModel(nn.Module):
    def __init__(self,size=64,pretrained=False,dropout=.3,backbone=None):
        super().__init__();self.scale=nn.Parameter(torch.ones(1));self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(6,1))
    def forward(self,cc,mlo):
        return self.head(torch.cat([cc.mean((2,3))*self.scale,mlo.mean((2,3))*self.scale],dim=1)).squeeze(1)
    def freeze_backbone(self,frozen):self.scale.requires_grad=not frozen


def test_resume_restores_states_history_and_larger_epoch_checkpoint(synthetic_root,tmp_path,monkeypatch):
    monkeypatch.setattr(train,'create_model',TinyResumeModel)
    base=['--size','64','--batch-size','4','--no-pretrained','--no-augmentations','--freeze-epochs','0',
          '--checkpoint-every','0','--bootstrap-iterations','10','--threshold-strategy','fixed']
    args=parse_args(base+['--output-dir',str(tmp_path/'interrupted'),'--epochs','1'])
    _,pairs=build_metadata(synthetic_root,tmp_path/'prepared',split_config=args)
    train.seed_everything(args.seed);train.train_model(pairs,args,torch.device('cpu'))
    original=pd.read_csv(args.output_dir/'history.csv')
    with pytest.warns(RuntimeWarning,match='pickle'):
        first=train.trusted_checkpoint_load(args.output_dir/'last.pt')
    for field in ('optimizer','scheduler','scaler','rng_state','train_loader_generator_state','val_loader_generator_state'):
        assert field in first
    resumed=parse_args(base+['--output-dir',str(args.output_dir),'--epochs','3','--resume',str(args.output_dir/'last.pt')])
    train.train_model(pairs,resumed,torch.device('cpu'))
    history=pd.read_csv(args.output_dir/'history.csv')
    assert history.epoch.tolist()==[1,2,3]
    pd.testing.assert_frame_equal(history.iloc[:1].reset_index(drop=True),original)
    uninterrupted=parse_args(base+['--output-dir',str(tmp_path/'uninterrupted'),'--epochs','3'])
    train.seed_everything(uninterrupted.seed);train.train_model(pairs,uninterrupted,torch.device('cpu'))
    with pytest.warns(RuntimeWarning):
        actual=train.trusted_checkpoint_load(args.output_dir/'last.pt')
    with pytest.warns(RuntimeWarning):
        expected=train.trusted_checkpoint_load(uninterrupted.output_dir/'last.pt')
    assert all(torch.equal(actual['model'][key],expected['model'][key]) for key in actual['model'])
    assert actual['history']==expected['history']
    no_op=parse_args(base+['--output-dir',str(args.output_dir),'--epochs','1','--resume',str(args.output_dir/'last.pt')])
    before=(args.output_dir/'last.pt').read_bytes()
    train.train_model(pairs,no_op,torch.device('cpu'))
    assert (args.output_dir/'last.pt').read_bytes()==before
    assert pd.read_csv(args.output_dir/'history.csv').epoch.tolist()==[1,2,3]


def test_real_end_to_end_all_holdout_cli_modes_and_checkpoint(synthetic_root,tmp_path):
    common=['--size','64','--batch-size','4','--no-pretrained','--no-augmentations','--freeze-epochs','0',
            '--checkpoint-every','0','--bootstrap-iterations','10','--threshold-strategy','fixed',
            '--gradcam-enabled','--gradcam-examples','1','--device','cpu','--output-dir',str(tmp_path/'e2e')]
    train.main(common+['--mode','prepare','--data-root',str(synthetic_root)])
    pairs=load_metadata_pairs(tmp_path/'e2e'/'metadata_pairs.csv',check_files=True)
    train.main(common+['--mode','sanity'])
    train.main(common+['--mode','train','--epochs','1'])
    train.main(common+['--mode','evaluate'])
    metrics=json.loads((tmp_path/'e2e'/'metrics.json').read_text())
    assert metrics['final_estimate'] and metrics['evaluation_role']=='test'
    train.main(common+['--mode','predict','--pair-id',pairs.iloc[0].pair_id])
    train.main(common+['--mode','predict','--cc-path',pairs.iloc[0].cc_dicom_path,'--mlo-path',pairs.iloc[0].mlo_dicom_path])
    output=pd.read_csv(tmp_path/'e2e'/'single_prediction.csv')
    assert len(output)==1 and np.isnan(output.iloc[0].label)
    assert {'probability','prediction','threshold','raw_probability','probability_kind'} <= set(output)
    assert (tmp_path/'e2e'/'best.pt').is_file() and (tmp_path/'e2e'/'last.pt').is_file()
    assert (tmp_path/'e2e'/'reliability_diagram.png').is_file()
    assert list((tmp_path/'e2e'/'heatmaps').glob('*.npz'))
    heatmap_file=next((tmp_path/'e2e'/'heatmaps').glob('*.npz'))
    with np.load(heatmap_file,allow_pickle=False) as heatmap:
        assert heatmap['heatmap'].shape==(80,50) and heatmap['model_heatmap'].shape==(64,64)
    assert list((tmp_path/'e2e'/'model_inputs').glob('*_model_heatmap.png'))
    train.main(common+['--mode','crossval','--folds','2','--epochs','1'])
    oof=pd.read_csv(tmp_path/'e2e'/'crossval'/'oof_predictions.csv')
    assert len(oof)==20 and oof.pair_id.nunique()==20
    assert oof.groupby('patient_id').fold.nunique().eq(1).all()
    assert set(oof.split)=={'test'} and set(oof.evaluation_role)=={'outer_test'}
    assert (tmp_path/'e2e'/'crossval'/'fold_1'/'metadata_pairs_development.csv').is_file()


def test_dicom_cannot_be_reused_between_views_without_reason(pair):
    pair.loc[0,'mlo_dicom_path']=pair.loc[0,'cc_dicom_path']
    with pytest.raises(ValueError,match='Isti DICOM'):
        validate_metadata_pairs(pair)


def test_breast_label_must_match_projection_birads(pair):
    pair['cc_birads']='2';pair['mlo_birads']='3'
    with pytest.raises(ValueError,match='Breast-level labela'):
        validate_metadata_pairs(pair)


class HookCountModel(nn.Module):
    gradcam_layout='NCHW'
    def __init__(self,calls=2):
        super().__init__();self.target=nn.Conv2d(3,2,1);self.calls=calls
    @property
    def gradcam_target(self):return self.target
    def forward(self,cc,mlo):
        outputs=[self.target(cc)]
        if self.calls>=2:outputs.append(self.target(mlo))
        if self.calls==3:outputs.append(self.target(cc))
        return sum(tensor.mean((1,2,3)) for tensor in outputs)


@pytest.mark.parametrize('calls',[1,3])
def test_gradcam_requires_exactly_two_target_hook_calls(calls):
    with pytest.raises(RuntimeError,match='tačno dva puta'):
        paired_gradcam(HookCountModel(calls),torch.rand(1,3,8,8),torch.rand(1,3,8,8))


def test_gradcam_negative_target_and_hook_cleanup():
    model=HookCountModel().train()
    cc,mlo=torch.rand(1,3,8,8),torch.rand(1,3,8,8)
    cams=paired_gradcam(model,cc,mlo,target_class=0)
    assert all(cam.shape==(8,8) and np.isfinite(cam).all() for cam in cams)
    assert model.training and not model.target._forward_hooks
    with pytest.raises(ValueError,match='target_class'):
        paired_gradcam(model,cc,mlo,target_class=2)


def test_cached_inputs_equal_uncached_inputs_and_keep_geometry(pair,tmp_path):
    uncached=data.load_prepared_view(pair.cc_dicom_path[0],'',64,return_geometry=True)
    cached=data.load_prepared_view(pair.cc_dicom_path[0],'',64,return_geometry=True,cache_dir=tmp_path/'cache')
    reloaded=data.load_prepared_view(pair.cc_dicom_path[0],'',64,return_geometry=True,cache_dir=tmp_path/'cache')
    for a,b,c in zip(uncached[:2],cached[:2],reloaded[:2]):
        assert np.array_equal(a,b) and np.array_equal(b,c)
    assert uncached[2]==cached[2]==reloaded[2]


def test_temperature_training_uses_disjoint_calibration_patients(synthetic_root,tmp_path,monkeypatch):
    monkeypatch.setattr(train,'create_model',TinyResumeModel)
    args=parse_args(['--size','64','--epochs','1','--batch-size','4','--no-pretrained','--no-augmentations',
                    '--calibration-method','temperature','--checkpoint-every','0','--output-dir',str(tmp_path/'calibration')])
    _,pairs=build_metadata(synthetic_root,tmp_path/'prepared_calibration',split_config=args)
    audit=split_audit(pairs)
    assert 'calibration' in audit['splits'] and all(not ids for ids in audit['intersections'].values())
    train.seed_everything(args.seed);train.train_model(pairs,args,torch.device('cpu'))
    with pytest.warns(RuntimeWarning):
        checkpoint=train.trusted_checkpoint_load(args.output_dir/'best.pt')
    assert checkpoint['calibration']['method']=='temperature'
    assert checkpoint['calibration']['source']=='independent_calibration_holdout'
    assert checkpoint['threshold_selection']['source']=='independent_threshold_holdout'


def test_direct_anonymous_dates_verified_from_existing_table(synthetic_root):
    cc=synthetic_root/'100_p0_MG_L_CC_ANON.dcm';mlo=synthetic_root/'101_p0_MG_L_MLO_ANON.dcm'
    for path in (cc,mlo):
        ds=pydicom.dcmread(path);del ds.AcquisitionDate;ds.save_as(path,enforce_file_format=True)
    table=pd.read_csv(synthetic_root/'INbreast.csv')
    table.loc[table['File Name']==101,'Acquisition date']=20200201
    table.to_csv(synthetic_root/'INbreast.csv',index=False)
    args=parse_args(['--cc-path',str(cc),'--mlo-path',str(mlo),'--size','64'])
    with pytest.raises(ValueError,match='acquisition date'):
        train.direct_prediction_frame(args)
    table.loc[table['File Name']==101,'Acquisition date']=20200101
    table.to_csv(synthetic_root/'INbreast.csv',index=False)
    frame=train.direct_prediction_frame(args)
    assert frame.iloc[0].acquisition_date=='20200101' and np.isnan(frame.iloc[0].label)
    assert frame.iloc[0].acquisition_date_source=='dataset_table|dataset_table'


def test_temperature_scaling_uses_original_logits_even_when_sigmoid_saturates():
    calibration={'applied':True,'temperature':10.}
    calibrated=apply_calibration([0.,1.],calibration,logits=[-40.,40.])
    assert calibrated[0]==pytest.approx(1/(1+np.exp(4.)))
    assert calibrated[1]==pytest.approx(1/(1+np.exp(-4.)))
    fitted=fit_temperature([0,1],[1.,0.],logits=[40.,-40.])
    assert fitted['logit_source']=='model_logits' and fitted['applied']


def test_direct_generic_filenames_use_verified_dicom_headers(pair,tmp_path):
    cc=Path(pair.cc_dicom_path[0]).rename(tmp_path/'hospital_exam_L_CC.dcm')
    mlo=Path(pair.mlo_dicom_path[0]).rename(tmp_path/'hospital_exam_L_MLO.dcm')
    args=parse_args(['--cc-path',str(cc),'--mlo-path',str(mlo),'--size','64'])
    frame=train.direct_prediction_frame(args)
    assert frame.iloc[0].patient_id=='p0' and frame.iloc[0].acquisition_date=='20200101'


def test_nonconstant_gradcam_equivalent_explicit_tensor_layouts():
    torch.manual_seed(3)
    activation=torch.rand(1,5,7,3)
    gradient=torch.rand_like(activation)
    nhwc=_to_cam(activation,gradient,(20,28),'NHWC')
    nchw=_to_cam(activation.permute(0,3,1,2),gradient.permute(0,3,1,2),(20,28),'NCHW')
    assert np.allclose(nhwc,nchw,atol=1e-6) and nhwc.max()==pytest.approx(1.)


def test_resume_older_checkpoint_to_fresh_directory_cannot_copy_newer_last(synthetic_root,tmp_path,monkeypatch):
    monkeypatch.setattr(train,'create_model',TinyResumeModel)
    base=['--size','64','--batch-size','4','--no-pretrained','--no-augmentations','--freeze-epochs','0','--checkpoint-every','0']
    args=parse_args(base+['--output-dir',str(tmp_path/'source'),'--epochs','1'])
    _,pairs=build_metadata(synthetic_root,tmp_path/'prepared',split_config=args)
    train.seed_everything(args.seed);train.train_model(pairs,args,torch.device('cpu'))
    with pytest.warns(RuntimeWarning):
        early=train.trusted_checkpoint_load(args.output_dir/'last.pt')
    torch.save(early,args.output_dir/'early.pt')
    with pytest.warns(RuntimeWarning):
        later=train.trusted_checkpoint_load(args.output_dir/'last.pt')
    later['epoch']=3
    torch.save(later,args.output_dir/'last.pt')
    restored=parse_args(base+['--output-dir',str(tmp_path/'branch'),'--epochs','0','--resume',str(args.output_dir/'early.pt')])
    train.train_model(pairs,restored,torch.device('cpu'))
    with pytest.warns(RuntimeWarning):
        result=train.trusted_checkpoint_load(restored.output_dir/'last.pt')
    assert result['epoch']==1 and result['history']==early['history']
    assert all(torch.equal(result['model'][key],early['model'][key]) for key in early['model'])
