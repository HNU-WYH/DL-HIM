from .deeponet import DeepONet1d
from .fns import FNS1d
from box import Box


def create_no(config: Box):
    ndim = config.problem.n_dim
    operator_type = config.training.operator_type

    if ndim == 1:
        if operator_type == "DeepONet":
            return DeepONet1d(config)
        elif operator_type == "FNS":
            return FNS1d(config)
        elif operator_type == "MIONet":
            return NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
