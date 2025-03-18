#!/usr/bin/env python3

from typing import Union, Any, List
from pathlib import Path
import numpy as np

from .data_parser_interface import Data_Parser_Interface

class KITTI_Velodyne_Dataset_Parser(Data_Parser_Interface):
    def __init__(self)->None:
        super().__init__()
        self._total_clouds: int = 0

    def parse_dataset_from(self, dataset_file: Union[str, Path])->Any:
        '''Load and parse a velodyne binary file.'''
        frame = np.fromfile(dataset_file, dtype=np.float32)
        return frame.reshape((-1, 4))

    def parse_in_batch(self, dataset_dir: Union[str, Path])->List[Any]:
        clouds = []
        for bin_file in dataset_dir:
            if Path(bin_file).suffix != '.bin':
                continue

            clouds.append( self.parse_dataset_from(bin_file) )
        self._total_clouds = len(clouds)
        
        return clouds

    def __len__(self)->int:
        return self._total_clouds
