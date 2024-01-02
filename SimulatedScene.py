import json
import torch
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class SimulatedScene:
    Name: str = ""
    Description: str = ""
    BoundingX: Tuple[float, float] =  (-10, 10)
    BoundingY: Tuple[float, float] = (-10, 10)
    BoundingZ: Tuple[float, float] = (-10, 10)
    Floor: float = -3.0
    Gravity: Tuple[float, float, float] = (0, 0, 9.8)
    dt: float = 0.01
    Steps: int = 50
    NewtonIts: int = 10
    LSIts: int = 50
    NumCubaturePts: int = 2000
    penalty_spring_fixed_weight: float = 10000.0
    penalty_spring_moving_weight: float = 0.0
    penalty_spring_floor_weight: float = 10000.0
    HessianSPDFix: bool = False
    BarrierInitStiffness: float = 1.0
    BarrierDec: float = 1.0
    BarrierIts: int = 1
    BarrierUpdate_code: str = "unused"
    SimplicitObjects: List = field(default_factory=list)
    CollisionObjects: List = field(default_factory=list)

    def add_object(self, name: str, position_delta: Tuple[float, float, float],
                   fixed_bc: torch.Tensor, moving_bc: torch.Tensor):
        self.SimplicitObjects.append({
                "Name": name,
                "PositionDelta": position_delta,
                "ym": "empty for now",
                "density": "empty for now",
                "SetFixedBC": fixed_bc,
                "SetMovingBC": moving_bc,
                "MoveBC": None, # "updated_vert_positions = None",
        })

    @classmethod
    def from_json(cls, scene_name: str):
        json_path = f'scenes/{scene_name}.json'
        scene_dict = json.loads(open(json_path, "r").read())
        scene = SimulatedScene(Name=scene_name, **scene_dict)
        # Turn BC fields off, as we use code fields instead: SetFixedBC_code, SetMovingBC_code, MoveBC_code
        for obj in scene.SimplicitObjects:
            obj["SetFixedBC"] = None
            obj["SetMovingBC"] = None
            obj["MoveBC"] = None
        return scene

        # self.SimplicitObjects.append({
        #     "Name": name,
        #     "PositionDelta": position_delta,
        #     "ym": "empty for now",
        #     "density": "empty for now",
        #     "SetFixedBC_code": "indices = torch.nonzero(X0[:,2] < -0.20, as_tuple=False).squeeze() #fix the whole bottom side of the shape",
        #     "SetMovingBC_code": "indices = torch.nonzero(X0[:,2] > 0.50, as_tuple=False).squeeze() #move the whole upper side of the shape",
        #     "MoveBC_code": "updated_vert_positions = None",
        #     "Codes": {}
        # }
        # )
