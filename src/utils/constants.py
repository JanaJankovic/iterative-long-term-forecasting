from typing import Literal

EncoderType = Literal["mlp", "rnn"]
DecoderType = Literal["mlp", "rnn"]
LatentFeatureMode = Literal["last", "mean", "flatten"]
RegressorName = Literal[
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "xgboost",
]
