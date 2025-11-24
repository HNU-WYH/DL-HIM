from .deeponet import DeepONet1d
from .fns import FNS1d
from box import Box


def create_no(config: Box):
    ndim = config.problem.n_dim
    operator_type = config.training.operator_type.lower()

    if ndim == 1:
        if operator_type == "deeponet":
            return DeepONet1d(config)
        elif operator_type == "fns":
            return FNS1d(config)
        elif operator_type == "mionet":
            return NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
