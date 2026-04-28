import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.cfg_util import load_config
from src.data_generation.generator2d import DataGenerator2d, TestDataGenerator

config = load_config("diffusion2d*")
DataGenerator2d(config).save(True)
TestDataGenerator(config).save(True)
