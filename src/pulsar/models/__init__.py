"""Model implementations: ANN, BNN, SNN, ANN-to-SNN, Pulsar."""
from .ann_baseline import MLPBaseline, CNNBaseline
from .bnn_baseline import BNNMLP, BNNCNN
from .snn_baseline import SNNMLP, SNNCNN
from .ann_to_snn import ANNtoSNNMLP, ANNtoSNNCNN
from .pulsar_v1 import PulsarMLP, PulsarCNN

__all__ = [
    "MLPBaseline",
    "CNNBaseline",
    "BNNMLP",
    "BNNCNN",
    "SNNMLP",
    "SNNCNN",
    "ANNtoSNNMLP",
    "ANNtoSNNCNN",
    "PulsarMLP",
    "PulsarCNN",
]

MODEL_REGISTRY = {
    "ann_mlp": MLPBaseline,
    "ann_cnn": CNNBaseline,
    "bnn_mlp": BNNMLP,
    "bnn_cnn": BNNCNN,
    "snn_mlp": SNNMLP,
    "snn_cnn": SNNCNN,
    "ann_to_snn_mlp": ANNtoSNNMLP,
    "ann_to_snn_cnn": ANNtoSNNCNN,
    "pulsar_mlp": PulsarMLP,
    "pulsar_cnn": PulsarCNN,
}
