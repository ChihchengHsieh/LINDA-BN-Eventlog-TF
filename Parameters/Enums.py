from enum import Enum


class SelectableDatasets(Enum):

    #############
    # PM
    #############

    BPI2012 = "BPI2012"
    Helpdesk = "Helpdesk"
    BPI2012WithResource = "BPI2012WithResource"

    #############
    # Medical
    #############
    Diabetes = "Diabetes"
    BreastCancer = "BreastCancer"


class SelectableLoss(Enum):
    CrossEntropy = "CrossEntropy"
    HingeLoss = "HingeLoss"
    BCE = "BCE"


class SelectableModels(Enum):
    BaseLineLSTMModel = "BaseLineLSTMModel"
    BaselineLSTMWithResource = "BaselineLSTMWithResource"
    BaseNNModel = "BaseNNModel"


class SelectableOptimizer(Enum):
    Adam = "Adam"
    SGD = "SGD"
    RMSprop = "RMSprop"


class SelectableLrScheduler(Enum):
    StepScheduler = "StepScheduler"
    ExponentialDecay= "ExponentialDecay"
    NotUsing = "NotUsing"

class PreprocessedDfType(Enum):
    Pickle = "Pickle"


class BPI2012ActivityType(Enum):
    O = "O"
    A = "A"
    W = "W"


class NumericalPermutationStrategies(Enum):
    Cube_All_Dim_Normal = "Cube_All_Dim_Normal"
    Cube_All_Dim_Uniform = "Cube_All_Dim_Uniform"
    Single_Feature_Unifrom = "Single_Feature_Unifrom"
    Ball_All_Dim_Uniform = "Ball_All_Dim_Uniform"
    Fix_Step = "Fix_Step"


class TracePermutationStrategies(Enum):
    SingleStepReplace = "SingleStepReplace"
    SampleFromEmbeddingDistance = "SampleFromEmbeddingDistance"
