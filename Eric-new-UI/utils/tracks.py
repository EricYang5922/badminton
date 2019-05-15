import numpy as np 

class Track_Set():
    def __init__(self, curve_list):
        self.curve_list = curve_list

    #player 1代表左方 2代表右方
    def select(self, player = None, track_type = None):
        result_list = []
        if player is not None:
            player = 2 * player - 3
        for no, track in enumerate(self.curve_list):
            if player is not None:
                if track.direction != player:
                    continue
            if track_type is not None:
                if track.type != track_type:
                    continue
            result_list.append(no)
        return result_list

    def get_curve_list(self, index_list):
        result_list = []
        for index in index_list:
            result_list.append(self.curve_list[index])
        return result_list
