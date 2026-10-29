from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class GameData:
    game_id: Optional[int] = None
    date: str = ""
    video: str = ""

@dataclass
class PlayData :
    play_id: Optional[int] = None
    game_id: int = 0
    timestamp: str = ""
    formation: str ="Unknown" 

@dataclass
class PlayerData:
    tracking_id: Optional[int] = None
    jersey_number: Optional[int] = None
    player_role: str = ""
    team_side : str = ""

@dataclass
class TrackingPoint:
    play_id: int
    tracking_id: int
    frame_id: int
    x_coord: float 
    y_coord: float

    def to_tuple(self):
        return (self.play_id, self.tracking_id, self.frame_id, self.x_coord, self.y_coord)

@dataclass
class RouteData:
    play_id: int
    tracking_id: int
    route_type: str = "Unknown"