from typing import List
from dataclasses import dataclass, field
from Parameters.Enums import (
    BPI2012ActivityType
)


@dataclass
class BPI2012Setting(object):
    file_path: str = "./datasets/event_logs/BPI_Challenge_2012.xes"

    preprocessed_folder_path: str = "./datasets/preprocessed/BPI_Challenge_2012_with_resource"

    include_types: List[BPI2012ActivityType] = field(
        default_factory=lambda: [BPI2012ActivityType.A, BPI2012ActivityType.O, BPI2012ActivityType.W])

    include_complete_only: bool = True

    def __post_init__(self):
        self.include_types = [BPI2012ActivityType[t] if type(
            t) == str else t for t in self.include_types]
