from .base import NeuralOperatorBase
from .deeponet import DeepONet1d
from .fns import FNS1d, FNS2d
from box import Box


def create_no(config: Box):
    ndim          = config.problem.n_dim
    operator_type = config.training.operator_type.lower()

    if ndim == 1:
        if operator_type == "deeponet":
            return DeepONet1d(config)
        elif operator_type == "fns":
            return FNS1d(config)
        elif operator_type == "mionet":
            raise NotImplementedError("MIONet is not yet implemented.")
        else:
            raise NotImplementedError(f"Unknown operator: {operator_type}")
    elif ndim == 2:
        if operator_type == "fns":
            return FNS2d(config)
        else:
            raise NotImplementedError(f"ndim=2 operator '{operator_type}' not implemented.")
    else:
        raise NotImplementedError(f"ndim={ndim} not supported.")
