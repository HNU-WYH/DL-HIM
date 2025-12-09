from src.utils.cfg_util import load_config
from src.data_generation.generator1d import DataGenerator1d, TestDataGenerator

config = load_config("diffusion1d*")
DataGenerator1d(config).save()
TestDataGenerator(config).save()
