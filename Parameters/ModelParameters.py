from dataclasses import dataclass, field

@dataclass
class TwoLayerLSTMPredNextWithResourceModelParameters(object):
    activity_embedding_dim: int = 32
    resource_embedding_dim: int = 128
    lstm_hidden: int = 64
    dense_dim: int = 64
    dropout: float = 0.1


@dataclass
class TwoLayerLSTMScenarioCfWithResourceModelParameters(object):
    activity_embedding_dim: int = 32
    resource_embedding_dim: int = 128
    lstm_hidden: int = 64
    dense_dim: int = 64
    dropout: float = 0.1
