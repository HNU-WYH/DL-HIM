from src.utils.cfg_util import load_config
from src.data_generation import create_generator, create_test_generator


config = load_config("diffusion1d*")
# config = load_config("diffusion2d*")
# config = load_config("helmholtz1d*")

gen = create_generator(config)
gen.save()

test_gen = create_test_generator(config)
test_gen.save()
