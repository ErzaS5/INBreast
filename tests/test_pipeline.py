from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from data import image_key, parse_label
from evaluate import classification_metrics, dice_iou
from preprocessing import prepare
from train import score_at_05


def test_label_parser():
    assert parse_label("BI-RADS 3") == 0
    assert parse_label("4a") == 1
    assert parse_label(6) == 1
    assert parse_label(0) is None
    assert parse_label("unknown") is None


def test_image_key():
    assert image_key(Path("20586908_x_MG_R_CC_ANON.dcm")) == "20586908"


def test_prepare_preserves_roi():
    image = np.zeros((100, 50), np.float32); image[10:90, 5:45] = .8
    mask = np.zeros_like(image, np.uint8); mask[12:15, 6:9] = 1
    resized, resized_mask = prepare(image, mask, 64)
    assert resized.shape == resized_mask.shape == (64, 64)
    assert resized_mask.sum() > 0
    assert set(np.unique(resized_mask)) <= {0, 1}


def test_metrics_shapes():
    metrics = classification_metrics([0, 1], [.1, .9])
    assert metrics["f1"] == 1
    assert np.asarray(metrics["confusion_matrix"]).shape == (2, 2)
    assert dice_iou(np.ones((2, 2)), np.ones((2, 2))) == (1., 1.)


def test_train_validation_score_is_bounded():
    assert score_at_05([0, 1], [.1, .9]) == 1.
    assert 0. <= score_at_05([0, 1], [.9, .1]) <= 1.
