"""Reproducible plots, prediction tables and structured breast-level error analysis."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from evaluate import calibration_metrics, classification_metrics_from_predictions


def write_json(path: Path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple,list)):
            return [clean(v) for v in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, (float,np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, Path):
            return str(item)
        return item
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def prediction_artifacts(rows, output, bins=10, metadata=None):
    output.mkdir(parents=True,exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'predictions.csv',index=False)
    y = frame.true_label.to_numpy(int)
    p = frame.probability.to_numpy(float)
    pred = frame.prediction.to_numpy(int)
    raw = frame.raw_probability.to_numpy(float) if 'raw_probability' in frame else p
    metrics = classification_metrics_from_predictions(y,pred,p,bins)
    matrix = np.asarray(metrics['confusion_matrix'])
    fig,axis=plt.subplots(figsize=(4,4))
    axis.imshow(matrix,cmap='Blues')
    for (r,c),n in np.ndenumerate(matrix):
        axis.text(c,r,str(n),ha='center',va='center')
    axis.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted breast label',ylabel='True breast label')
    fig.tight_layout();fig.savefig(output/'confusion_matrix.png',dpi=140);plt.close(fig)
    if len(np.unique(y))==2:
        fpr,tpr,thresholds=roc_curve(y,p)
        precision,recall,pr_thresholds=precision_recall_curve(y,p)
        pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thresholds}).to_csv(output/'roc_curve.csv',index=False)
        pd.DataFrame({'precision':precision,'recall':recall,'threshold':np.r_[pr_thresholds,np.nan]}).to_csv(output/'precision_recall_curve.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(fpr,tpr);axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='ROC')
        axes[1].plot(recall,precision);axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision-recall')
        fig.tight_layout();fig.savefig(output/'classification_curves.png',dpi=140);plt.close(fig)
    before,after=calibration_metrics(y,raw,bins),calibration_metrics(y,p,bins)
    fig,axis=plt.subplots(figsize=(5,5))
    axis.plot([0,1],[0,1],'--',color='gray')
    for name,cal in (('raw',before),('reported',after)):
        occupied=[b for b in cal['bins'] if b['count']]
        axis.plot([b['mean_probability'] for b in occupied],[b['positive_fraction'] for b in occupied],'o-',label=name)
    axis.set(xlim=(0,1),ylim=(0,1),xlabel='Mean probability',ylabel='Observed positive fraction',title='Reliability diagram')
    axis.legend();fig.tight_layout();fig.savefig(output/'reliability_diagram.png',dpi=140);plt.close(fig)
    write_json(output/'calibration_evaluation.json',{'before':before,'after':after,'probability_kind':frame.probability_kind.value_counts().to_dict() if 'probability_kind' in frame else {'raw':len(frame)}})
    error_analysis(frame,output,metadata)


def error_analysis(frame,output,metadata=None):
    frame=frame.copy()
    frame['category']=np.select([(frame.true_label==1)&(frame.prediction==0),(frame.true_label==0)&(frame.prediction==1),
                                (frame.true_label==1)&(frame.prediction==1)],['FN','FP','TP'],default='TN')
    frame['confidence']=np.where(frame.prediction==1,frame.probability,1-frame.probability)
    for category in ('FN','FP','TP','TN'):
        frame[frame.category==category].sort_values('confidence',ascending=False).to_csv(output/f'{category.lower()}_predictions.csv',index=False)
    frame[frame.category.isin(['FN','FP'])].sort_values('confidence',ascending=False).head(20).to_csv(output/'most_confident_errors.csv',index=False)
    frame.sort_values('confidence').head(20).to_csv(output/'least_confident_predictions.csv',index=False)
    frame.assign(uncertainty=np.abs(frame.probability-frame.threshold)).sort_values('uncertainty').head(20).to_csv(output/'least_certain_predictions.csv',index=False)
    if metadata is not None and 'acquisition_date' in metadata:
        counts=metadata.groupby('patient_id').acquisition_date.nunique()
        frame['patient_exam_count']=frame.patient_id.map(counts)
        frame['multiple_exams']=frame.patient_exam_count.gt(1)
    if {'cc_candidate_count','mlo_candidate_count'} <= set(frame):
        frame['repeated_exposure']=pd.to_numeric(frame.cc_candidate_count).gt(1)|pd.to_numeric(frame.mlo_candidate_count).gt(1)
    if {'cc_rows','cc_columns','mlo_rows','mlo_columns'} <= set(frame):
        frame['resolution']=frame[['cc_rows','cc_columns','mlo_rows','mlo_columns']].astype(str).agg('x'.join,axis=1)
    if {'cc_has_roi','mlo_has_roi'} <= set(frame):
        frame['has_roi']=pd.to_numeric(frame.cc_has_roi).gt(0)|pd.to_numeric(frame.mlo_has_roi).gt(0)
    groups={}
    for column in ('birads','side','resolution','has_roi','multiple_exams','repeated_exposure'):
        if column in frame:
            groups[column]={str(key):classification_metrics_from_predictions(part.true_label,part.prediction,part.probability)
                            for key,part in frame.groupby(column,dropna=False)}
    write_json(output/'error_analysis.json',{'categories':frame.category.value_counts().to_dict(),'subgroups':groups,
        'visual_review_status':'not yet reviewed; inspect original-image overlays for text, edges, padding, scanner artifacts, pectoral muscle and crop shortcuts',
        'limitation':'Grad-CAM overlap with ROI is not evidence of clinically correct learned features.'})
