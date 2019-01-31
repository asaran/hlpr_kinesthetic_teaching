import random
import numpy as np
import birl
import active_utils as autils

import active_var_complexreward as active_var
import sys

from copy import deepcopy
import pickle as pkl


class BIRLPlacement():
    def __init__(self):
        # print('created placement object')
        self.birl_params = None

    def load_learned_params():
        with open('birl_params.pkl', 'rb') as handle:
            self.birl_params = pickle.load(handle)

    def get_place_loc(config):
        map_params = birl_params.get_map_params()
        self.place_loc = active_var.get_best_placement(config, map_params)


if __name__=="__main__":
    birl_obj = BIRLPlacement()
    birl_obj.load_learned_params()
    bowl_loc = [x,y]
    plate_loc = [x,y]
    obj_centers = np.array([plate_loc,bowl_loc])
    birl_obj.get_place_loc(obj_centers)
