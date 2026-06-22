"""HDB game-master prefabs used by the market simulation."""

from . import hdb_coordinator_gm
from . import hdb_initializer_gm

CoordinatorGameMaster = hdb_coordinator_gm.GameMaster
InitialiserGameMaster = hdb_initializer_gm.InitialiserGameMaster

__all__ = [
    'hdb_coordinator_gm',
    'hdb_initializer_gm',
    'CoordinatorGameMaster',
    'InitialiserGameMaster',
]
