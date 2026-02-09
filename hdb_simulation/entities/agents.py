import dataclasses
from typing import Optional, List, Dict, Any

from concordia.concordia.typing import prefab as prefab_lib
from concordia.hdb_simulation.models.schemas import Flat

dataclass = dataclasses.dataclass

@dataclass
class BuyerAgent(prefab_lib.Prefab):
    # Standard metadata
    id: str
    name: str
    age: int
    occupation: str
    preferences: Dict[str, Any]
    budget: Dict[str, float]
    description: Optional[str] = None

    # TODO: build using data 
    def build(self,model):
        return 
    


@dataclass
class SellerAgent(prefab_lib.Prefab):
    # Standard metadata
    id: str
    name: str
    age: int
    occupation: str
    flat: Optional[Flat]
    expectations: Dict[str, float]
    description: Optional[str] = None

    # TODO: build using data
    def build(self,model): 
        return
    
