from .place_env import *

from gym.envs.registration import register

register(
	id = 'place_env-v0',
	entry_point = 'src.place_env.place_env:PlaceEnv'
)

register(
	id = 'place_colenv-v1',
	entry_point = 'src.place_env.place_env:PlaceColEnv'
)