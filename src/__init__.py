"""
src — CS4452 action recognition on THUMOS-14.

Public API
----------
from src.models import build_backbone, MeanPooling, AttentionPooling, ClassificationHead
from src.models.classifier import ActionRecognitionModel, TemporalDetectionHead
from src.dataset import THUMOSVideoDataset, THUMOSFeatureDataset, THUMOS14_CLASSES
from src.utils import get_logger, save_checkpoint, load_checkpoint, iou_1d
"""
