import json
from pathlib import Path


class ClimbingConfig:

    def __init__(self, path_to_config):
        self.data = {}

        with open(path_to_config) as f:
            self.data = json.load(f)

        self.assets = self.data["assets"]
        self.timestep_interval = self.data["timestep_interval"]
        self.timestep_per_action = self.data["timestep_per_action"]
        self.holds = self.data["holds"]
        self.stance_path = self.data["stance_path"]
        self.climber = self.data["climber"]
        self.surface = self.data["surface"]
        self.plane = self.data["ground_plane"]

        # Initialize hold_grid_mapping to prevent workers from crashing
        # before the CSV dynamic route loop runs in your environment's reset()
        self.hold_grid_mapping = {}

        # Link asset details down to the active scene components
        for key in self.holds:
            self.holds[key]["asset_data"] = self.assets[
                self.holds[key]["asset"]
            ]

        self.surface["asset_data"] = self.assets[self.surface["asset"]]
        self.plane["asset_data"] = self.assets[self.plane["asset"]]

        # Securely parse the secondary kilter configuration metadata if defined
        if self.data.get("path_to_kilter_config"):
            kilter_path = self.data["path_to_kilter_config"]
            with open(kilter_path) as f2:
                self.kilter_cfg = json.load(f2)
        else:
            self.kilter_cfg = None
