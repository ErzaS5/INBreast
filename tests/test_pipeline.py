from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[1]))
import data as data_module
import train as train_module
from data import (PairedINbreastDataset, acquisition_key, assign_patient_folds, image_key,
                  load_metadata_pairs, parse_label, select_pairing_views)
from evaluate import classification_metrics, classification_metrics_from_predictions, dice_iou
from gradcam import paired_gradcam, save_prediction_artifacts
from model import create_model
from preprocessing import add_poisson_noise, breast_bbox, load_xml_mask, normalize_dicom_pixels, prepare
from train import run_cross_validation, run_epoch


def test_label_parser():
    assert parse_label("BI-RADS 3") == 0
    assert parse_label("4a") == 1
    assert parse_label(6) == 1
    assert parse_label(0) is None
    assert parse_label("unknown") is None
    assert acquisition_key(200901.0) == "200901"


def test_repeated_view_selection_is_auditable():
    images = pd.DataFrame([
        {"patient_id": "p", "acquisition_date": "200901", "side": "R", "view": "CC", "image_id": "2", "label": 0, "has_roi": 1},
        {"patient_id": "p", "acquisition_date": "200901", "side": "R", "view": "CC", "image_id": "1", "label": 1, "has_roi": 1},
    ])
    selected, audit = select_pairing_views(images)
    assert selected.image_id.tolist() == ["1"]
    assert len(audit) == 2
    assert audit.selected_for_pairing.sum() == 1


def test_image_key():
    assert image_key(Path("20586908_x_MG_R_CC_ANON.dcm")) == "20586908"


def test_prepare_preserves_roi():
    image = np.zeros((100, 50), np.float32); image[10:90, 5:45] = .8
    mask = np.zeros_like(image, np.uint8); mask[12:15, 6:9] = 1
    resized, resized_mask = prepare(image, mask, 64)
    assert resized.shape == resized_mask.shape == (64, 64)
    assert resized_mask.sum() > 0
    assert set(np.unique(resized_mask)) <= {0, 1}


def test_crop_is_independent_of_ground_truth_mask():
    image = np.zeros((120, 80), np.float32); image[10:115, 8:70] = .5
    first_mask = np.zeros_like(image, np.uint8); first_mask[30:40, 20:30] = 1
    second_mask = np.zeros_like(image, np.uint8); second_mask[2:6, 74:78] = 1
    first_image, _ = prepare(image, first_mask, 64)
    second_image, _ = prepare(image, second_mask, 64)
    assert np.array_equal(first_image, second_image)


def test_intensity_crop_selects_large_breast_component():
    image = np.zeros((200, 120), np.float32)
    image[10:195, 5:105] = .35
    image[5:12, 112:118] = 1.0  # synthetic scanner label/artifact
    top, left, bottom, right = breast_bbox(image)
    assert top <= 10 and left <= 5
    assert bottom >= 190 and right >= 100


def test_poisson_noise_preserves_range_and_expected_signal():
    np.random.seed(7)
    image = np.full((128, 128), .5, np.float32)
    noisy = add_poisson_noise(image, peak=120)
    assert noisy.shape == image.shape
    assert noisy.dtype == np.float32
    assert 0 <= noisy.min() <= noisy.max() <= 1
    assert not np.array_equal(noisy, image)
    assert abs(float(noisy.mean()) - .5) < .02


def test_dicom_normalization_handles_padding_and_monochrome1():
    image = np.array([[0, 0], [100, 200]], dtype=np.float32)
    padding = image == 0
    normalized = normalize_dicom_pixels(image, padding, "MONOCHROME1", "test")
    assert normalized.dtype == np.float32
    assert np.all(normalized[padding] == 0)
    assert normalized[1, 0] > normalized[1, 1]


def test_metrics_shapes():
    metrics = classification_metrics([0, 1], [.1, .9])
    assert metrics["f1"] == 1
    assert np.asarray(metrics["confusion_matrix"]).shape == (2, 2)
    assert dice_iou(np.ones((2, 2)), np.ones((2, 2))) == (1., 1.)


def test_metrics_support_fold_specific_predictions():
    metrics = classification_metrics_from_predictions([0, 0, 1, 1], [0, 1, 1, 1], [.1, .7, .6, .9])
    assert metrics["f1"] > 0
    assert metrics["roc_auc"] > .5


def test_patient_folds_have_no_leakage_and_full_coverage():
    frame = pd.DataFrame([
        {"patient_id": f"p{patient}", "pair_id": f"p{patient}_{side}", "side": side, "label": patient % 2}
        for patient in range(10) for side in ("L", "R")
    ])
    folded = assign_patient_folds(frame, n_splits=5, seed=3)
    assert set(folded.fold) == {1, 2, 3, 4, 5}
    assert folded.groupby("patient_id").fold.nunique().max() == 1
    assert folded.groupby("fold").patient_id.nunique().eq(2).all()


def test_xml_polygon_is_loaded(tmp_path):
    import plistlib
    xml_path = tmp_path / "roi.xml"
    document = {"Images": [{"ROIs": [{"Point_px": ["(2, 2)", "(8, 2)", "(8, 8)", "(2, 8)"]}]}]}
    with xml_path.open("wb") as stream:
        plistlib.dump(document, stream)
    mask = load_xml_mask(xml_path, (12, 12))
    assert mask.dtype == np.uint8
    assert mask[5, 5] == 1
    assert mask.sum() > 20


def test_dataset_and_dataloader_shapes(monkeypatch):
    image = np.zeros((80, 50), np.float32); image[5:75, 4:46] = .7
    mask = np.zeros_like(image, np.uint8); mask[25:35, 20:30] = 1
    monkeypatch.setattr(data_module, "read_dicom", lambda _path: image.copy())
    monkeypatch.setattr(data_module, "load_xml_mask", lambda _path, _shape: mask.copy())
    frame = pd.DataFrame([{
        "pair_id": "patient_L", "patient_id": "patient", "side": "L", "label": 1,
        "cc_dicom_path": "cc.dcm", "cc_xml_path": "cc.xml", "cc_has_roi": 1,
        "mlo_dicom_path": "mlo.dcm", "mlo_xml_path": "mlo.xml", "mlo_has_roi": 1,
    }])
    batch = next(iter(DataLoader(PairedINbreastDataset(frame, size=64, training=False), batch_size=1)))
    assert batch["cc_image"].shape == batch["mlo_image"].shape == (1, 3, 64, 64)
    assert batch["cc_mask"].shape == batch["mlo_mask"].shape == (1, 1, 64, 64)
    assert batch["label"].tolist() == [1.]


def test_prepared_metadata_is_loaded_without_rebuilding(tmp_path):
    path = tmp_path / "metadata_pairs.csv"
    pd.DataFrame([{
        "pair_id": "p1_200901_L", "patient_id": "p1", "acquisition_date": "200901",
        "exam_id": "p1_200901", "metadata_schema_version": 2,
        "side": "L", "label": 1, "split": "val",
        "cc_dicom_path": "cc.dcm", "cc_xml_path": "", "cc_has_roi": 0,
        "mlo_dicom_path": "mlo.dcm", "mlo_xml_path": "", "mlo_has_roi": 0,
    }]).to_csv(path, index=False)
    loaded = load_metadata_pairs(path)
    assert loaded.pair_id.tolist() == ["p1_200901_L"]
    assert loaded.split.tolist() == ["val"]


def test_dataset_supports_unlabelled_direct_inference(monkeypatch):
    image = np.zeros((32, 32), np.float32)
    monkeypatch.setattr(data_module, "read_dicom", lambda _path: image.copy())
    frame = pd.DataFrame([{
        "pair_id": "direct", "patient_id": "", "side": "", "label": np.nan,
        "cc_dicom_path": "cc.dcm", "cc_xml_path": "", "cc_has_roi": 0,
        "mlo_dicom_path": "mlo.dcm", "mlo_xml_path": "", "mlo_has_roi": 0,
    }])
    item = PairedINbreastDataset(frame, size=32, training=False)[0]
    assert torch.isnan(item["label"])


class _ChannelLastLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, image):
        return image.permute(0, 2, 3, 1) * self.scale


class _TinyDualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = _ChannelLastLayer()
        self.classifier = nn.Linear(6, 1)

    @property
    def gradcam_target(self):
        return self.target

    def encode(self, image):
        return self.target(image).mean(dim=(1, 2))

    def forward(self, cc_image, mlo_image):
        return self.classifier(torch.cat([self.encode(cc_image), self.encode(mlo_image)], dim=1)).squeeze(1)


class _TinyPairDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"cc_image": torch.rand(3, 16, 16), "mlo_image": torch.rand(3, 16, 16),
                "label": torch.tensor(float(index % 2))}


def test_training_step_and_gradcam_smoke():
    model = _TinyDualModel()
    loader = DataLoader(_TinyPairDataset(), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loss, labels, probabilities = run_epoch(model, loader, nn.BCEWithLogitsLoss(), optimizer, scaler, torch.device("cpu"), True)
    assert np.isfinite(loss)
    assert len(labels) == len(probabilities) == 4
    cc, mlo = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    cc_cam, mlo_cam = paired_gradcam(model, cc, mlo)
    assert cc_cam.shape == mlo_cam.shape == (16, 16)
    assert 0 <= cc_cam.min() <= cc_cam.max() <= 1


def test_prediction_visualizations_are_saved(tmp_path):
    image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    heatmap = np.flipud(image).copy()
    mask = np.zeros((32, 32), np.uint8); mask[10:20, 12:22] = 1
    paths = save_prediction_artifacts(image, heatmap, mask, .5, tmp_path, "sample_CC", "sample")
    assert all(Path(path).is_file() for path in paths.values())


def test_real_swin_forward_and_gradcam():
    model = create_model(size=224, pretrained=False).eval()
    cc, mlo = torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(cc, mlo)
    assert logits.shape == (1,) and torch.isfinite(logits).all()
    cc_cam, mlo_cam = paired_gradcam(model, cc, mlo)
    assert cc_cam.shape == mlo_cam.shape == (224, 224)
    assert np.isfinite(cc_cam).all() and np.isfinite(mlo_cam).all()


@pytest.mark.parametrize("backbone", ["resnet18", "densenet121"])
def test_real_cnn_forward_and_gradcam(backbone):
    model = create_model(size=64, pretrained=False, backbone=backbone).eval()
    cc, mlo = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        logits = model(cc, mlo)
    assert logits.shape == (1,) and torch.isfinite(logits).all()
    cc_cam, mlo_cam = paired_gradcam(model, cc, mlo)
    assert cc_cam.shape == mlo_cam.shape == (64, 64)
    assert np.isfinite(cc_cam).all() and np.isfinite(mlo_cam).all()


def test_frozen_resnet_backbone_stays_in_eval_mode_during_head_training():
    model = create_model(size=64, pretrained=False, backbone="resnet18")
    model.freeze_backbone(True)
    model.train()
    assert not model.backbone.training
    assert model.classifier.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    model.freeze_backbone(False)
    model.train()
    assert model.backbone.training
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(module.training for module in model.backbone.modules()
                   if isinstance(module, nn.modules.batchnorm._BatchNorm))


def test_crossval_orchestrator_writes_complete_oof_results(tmp_path, monkeypatch):
    pairs = pd.DataFrame([
        {"patient_id": f"p{patient}", "pair_id": f"p{patient}_{side}", "side": side,
         "label": patient % 2, "split": "train"}
        # Independent checkpoint and threshold holdouts each need both patient strata.
        # The old 8-patient fixture only passed through a silent random split fallback.
        for patient in range(40) for side in ("L", "R")
    ])
    args = SimpleNamespace(output_dir=tmp_path, folds=2, seed=11, inner_val_fraction=.25,
                           bootstrap_iterations=0)

    monkeypatch.setattr(train_module, "train_model", lambda _pairs, _args, _device: None)
    monkeypatch.setattr(train_module, "load_checkpoint", lambda _args, _device: (object(), {"threshold": .5}))
    monkeypatch.setattr(train_module, "predict_frame", lambda _model, frame, _args, _device: [
        {"pair_id": row.pair_id, "patient_id": row.patient_id, "side": row.side,
         "true_label": int(row.label), "probability": .8 if row.label else .2}
        for row in frame.itertuples()
    ])
    monkeypatch.setattr(train_module, "collect_localization", lambda *_args, **_kwargs: ([], [], []))

    run_cross_validation(pairs, args, torch.device("cpu"))
    output = tmp_path / "crossval"
    oof = pd.read_csv(output / "oof_predictions.csv")
    assert len(oof) == len(pairs)
    assert oof.pair_id.nunique() == len(pairs)
    assert oof.groupby("patient_id").fold.nunique().max() == 1
    assert (output / "fold_metrics.csv").is_file()
    assert (output / "metrics.json").is_file()
