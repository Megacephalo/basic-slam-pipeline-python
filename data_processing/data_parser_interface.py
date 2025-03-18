#!/usr/bin/env python3

from typing import Union, Any, List
from pathlib import Path

class Data_Parser_Interface:
    def parse_dataset_from(self, dataset_file: Union[str, Path])->Any:
        '''Parse a dataset in the expected data format from the provided dataset_file'''
        return None
    
    def parse_in_batch(self, dataset_dir: Union[str, Path])->List[Any]:
        '''parse out multiple dataset deliverables from the same type of files under the provided directory'''
        return []
    
    def __len__(self)->int:
        '''Output the number of total parsed point clouds'''
        return 0