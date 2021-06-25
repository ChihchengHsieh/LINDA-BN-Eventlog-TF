from typing import List
from Utils.SaveUtils import get_json_dict
from dataclasses import dataclass, field
from Parameters.Enums import (
    SelectableDatasets,
    SelectableLoss,
    SelectableLrScheduler,
    SelectableModels,
    SelectableOptimizer,
    ActivityType
)

@dataclass
class BPI2012(object):
    BPI2012_include_types: List[ActivityType] = field(
        default_factory=lambda: [ActivityType.A, ActivityType.O, ActivityType.W])

    include_complete_only: bool = True

    def __post_init__(self):
        self.BPI2012_include_types = [ActivityType[t] if type(
            t) == str else t for t in self.BPI2012_include_types]

@dataclass
class OptimizerParameters(object):
    """
    It will be override once you have load_model and load_optimizer = True
    """
    ###### XES ######
    learning_rate: float = 0.0005
    l2: float = 0.0000000001

    ###### Medical ######
    # learning_rate: float = 0.005
    # l2: float = 0.001

    # Scheduler
    scheduler: SelectableLrScheduler = SelectableLrScheduler.StepScheduler
    lr_scheduler_step: int = 800
    lr_scheduler_gamma: float = 0.8
    SGD_momentum: float = 0.9

    def __post_init__(self):
        if (type(self.scheduler) == str):
            self.scheduler = SelectableLrScheduler[self.scheduler]

@dataclass
class BaseNNModelParams(object):
    hidden_dim: List[int] = field(default_factory=lambda: [8]*8)
    dropout: float = .2

@dataclass
class BaselineLSTMModelParameters(object):
    """
    It will be override once you have load_model
    """
    embedding_dim: int = 32  # 128
    lstm_hidden: int = 64  # 256
    dropout: float = 0.1
    num_lstm_layers: int = 1  # 2

@dataclass
class BaselineLSTMWithResourceparameters(object):
    activity_embedding_dim: int = 32
    resource_embedding_dim: int = 128
    lstm_hidden: int = 32  # 256
    dense_dim: int = 64
    dropout: float = 0.1

@dataclass
class TransformerParameters(object):
    num_layers: int = 4
    model_dim: int = 32
    feed_forward_dim = 64
    num_heads = 4
    dropout_rate: float = .1

@dataclass
class TrainingParameters(object):
    """
    Storing the parameters for controlling the training.
    """

    #########################
    # Load
    ########################

    # load_model_folder_path: str = "SavedModels/0.8318_BPI2012_BaseLineLSTMModel_2021-05-10 04:35:30.266388" # Set to None to not loading pre-trained model.
    # Set to None to not loading pre-trained model.
    load_model_folder_path: str = None
    load_optimizer: bool = True

    ######################################
    # Selectables
    #####################################
    dataset: SelectableDatasets = SelectableDatasets.BPI2012WithResource
    model: SelectableModels = SelectableModels.BaselineLSTMWithResource
    loss: SelectableLoss = SelectableLoss.CrossEntropy
    optimizer: SelectableOptimizer = SelectableOptimizer.Adam

    ######################################
    # Count
    ######################################
    stop_epoch: int = 5
    batch_size: int = 128 #128
    verbose_freq: int = 250  # in step
    run_validation_freq: int = 500  # in step

    ######################################
    # Dataset
    ######################################
    # Remaining will be used for validation.
    train_test_split_portion: List[float] = field(
        default_factory=lambda: [0.8, 0.1])
    dataset_split_seed: int = 12345

    ########################
    # Others
    ########################
    max_eos_predicted_length: int = 50
    plot_cm: bool = False

    bpi2012: BPI2012 = BPI2012()
    baselineLSTMModelParameters: BaselineLSTMModelParameters = BaselineLSTMModelParameters()
    optimizerParameters: OptimizerParameters = OptimizerParameters()
    baseNNModelParams: BaseNNModelParams = BaseNNModelParams()
    baselineLSTMWithResourceparameters: BaselineLSTMWithResourceparameters = BaselineLSTMWithResourceparameters()
    transformerParameters: TransformerParameters = TransformerParameters()

    def __post_init__(self):
        if (type(self.baselineLSTMModelParameters) == dict):
            self.baselineLSTMModelParameters = BaselineLSTMModelParameters(
                **self.baselineLSTMModelParameters)

        if (type(self.baselineLSTMWithResourceparameters) == dict):
            self.baselineLSTMWithResourceparameters = BaselineLSTMWithResourceparameters(
                **self.baselineLSTMWithResourceparameters)

        if (type(self.transformerParameters) == dict):
            self.transformerParameters = TransformerParameters(
                **self.transformerParameters)

        if (type(self.bpi2012) == dict):
            self.bpi2012 = BPI2012(**self.bpi2012)

        if (type(self.optimizerParameters) == dict):
            self.optimizerParameters = OptimizerParameters(
                **self.optimizerParameters)

        if (type(self.baseNNModelParams) == dict):
            self.baseNNModelParams = BaseNNModelParams(
                **self.baseNNModelParams)

        if (type(self.dataset) == str):
            self.dataset = SelectableDatasets[self.dataset]

        if (type(self.model) == str):
            self.model = SelectableModels[self.model]

        if (type(self.loss) == str):
            self.loss = SelectableLoss[self.loss]

        if (type(self.optimizer) == str):
            self.optimizer = SelectableOptimizer[self.optimizer]
