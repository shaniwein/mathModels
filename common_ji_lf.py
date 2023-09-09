import numpy as np
from enum import Enum
from collections import namedtuple

PlayerData = namedtuple("PlayerData", ["id", "ji_filenames", "follower_filenames", "follower_indexs"])

players_data = [
    PlayerData(id=33, ji_filenames=["a.player33_player36_ji.mat", "b.player33_player36_ji.mat", "player33_player34_ji.mat", "player33_player35_ji.mat"],
               follower_filenames=["player33_player34_lf_34_1.mat", "player33_player34_lf_34_2.mat"], follower_indexs = [0,0]),
    PlayerData(id=34, ji_filenames=["player33_player34_ji.mat", "player34_player35_ji.mat", "player34_player36_ji.mat"],
               follower_filenames=["player33_player34_lf_33_1.mat", "player35_player34_lf_35.mat"], follower_indexs = [1,0]),
    PlayerData(id=35, ji_filenames=["player33_player35_ji.mat", "player34_player35_ji.mat", "player35_player36_ji.mat"],
               follower_filenames=["player35_player36_lf_36.mat", "player35_player34_lf_34.mat"], follower_indexs =[0, 1]),
    PlayerData(id=36, ji_filenames=["a.player33_player36_ji.mat", "b.player33_player36_ji.mat", "player34_player36_ji.mat", "player35_player36_ji.mat"],
               follower_filenames=["player35_player36_lf_35.mat"], follower_indexs=[1])
]

marker_tags = [
    "HeadFront",
    "LElbowOut",
    "LHand2",
    "RElbowOut",
    "RHand2",
    "WaistBack",
    "LKneeOut",
    "RKneeOut",
]

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
    raise ValueError(f"label {wanted_label} not found in {qtm_obj.labels[ind]}")