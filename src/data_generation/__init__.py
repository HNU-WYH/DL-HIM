from box import Box

from src.data_generation.generator1d import DataGenerator1d, TestDataGenerator
from src.data_generation.generator2d import DataGenerator2d, TestDataGenerator2d


def create_generator(config: Box):
    """Factory: return train/val data generator based on config.problem.n_dim."""
    ndim = config.problem.n_dim
    if ndim == 1:
        return DataGenerator1d(config)
    elif ndim == 2:
        return DataGenerator2d(config)
    else:
        raise NotImplementedError(f"Data generation for n_dim={ndim} is not supported.")


def create_test_generator(config: Box, test_num=None):
    """Factory: return test data generator based on config.problem.n_dim."""
    ndim = config.problem.n_dim
    if ndim == 1:
        return TestDataGenerator(config, test_num)
    elif ndim == 2:
        return TestDataGenerator2d(config, test_num)
    else:
        raise NotImplementedError(f"Test data generation for n_dim={ndim} is not supported.")
