import numpy as np
from enum import Enum
from collections import namedtuple

PlayerData = namedtuple("PlayerData", ["id", "solo_filenames", "follower_filename", "follower_indexs"])

players_data = [
    PlayerData(id=33, solo_filenames=["player33_solo0001.mat", "player33_solo0002.mat", "player33_solo0003.mat"],
               follower_filename=["player33_player34_lf_34_1.mat", "player33_player34_lf_34_2.mat"], follower_indexs = [0,0]),
    PlayerData(id=34, solo_filenames=["player34_solo0001.mat", "player34_solo0002.mat", "player34_solo0003.mat"],
               follower_filename=["player33_player34_lf_33_1.mat", "player35_player34_lf_35.mat"], follower_indexs = [1,0]),
    PlayerData(id=35, solo_filenames=["player35_solo0001.mat", "player35_solo0002.mat", "player35_solo0003.mat"],
               follower_filename=["player35_player36_lf_36.mat", "player35_player34_lf_34.mat"], follower_indexs =[0, 1]),
    PlayerData(id=36, solo_filenames=["player36_solo0001.mat", "player36_solo0002.mat", "player36_solo0003.mat"],
               follower_filename=["player35_player36_lf_35.mat"], follower_indexs=[1])
]

marker_tags_to_solo_marker_id = {
    "HeadFront": 2,
    "LElbowOut": 7,
    "LHand2": 9,
    "RElbowOut": 12,
    "RHand2": 14,
    "WaistBack": 17,
    "LKneeOut": 21,
    "RKneeOut": 28
}

# solo_selected_markers = [2, 7, 9, 12, 14, 17, 21, 28]

def select_markers(qtm_obj, player, ind):
    """
    select only chosen markers from duet: HeadFront, LElbowOut, LHand2, RElbowOut, RHand2, WaistBack, LKneeOut, RKneeOut

    :return: indexs of selected markers
    """
    marker_labels = [f"player{player.id}_HeadFront",
                     f"player{player.id}_LElbowOut", f"player{player.id}_LHand2", f"player{player.id}_RElbowOut",
                     f"player{player.id}_RHand2", f"player{player.id}_WaistBack", f"player{player.id}_LKneeOut",
                     f"player{player.id}_RKneeOut"]
    selected_markers = []
    for i, label in np.ndenumerate(qtm_obj.labels[ind]):
        for wanted_label in marker_labels:
            if label == wanted_label:
                selected_markers.append(i[0])
    return selected_markers

def select_marker_by_tag(qtm_obj, player, ind, tag):
    wanted_label = f"player{player.id}_{tag}"
    for i, label in np.ndenumerate(qtm_obj.labels[ind]):
        if label == wanted_label:
            return i[0]
    raise ValueError(f"tag {tag} not found in {qtm_obj.labels[ind]}")