import os
import re
import numpy as np
from collections import namedtuple, defaultdict
from matplotlib import pyplot as plt
from qtmWrapper.qtm import QTM
from common import select_marker_by_tag, players_data, marker_tags_to_solo_marker_id
import smoothnessMeasurement

PlayerData = namedtuple("PlayerData", ["id", "solo_filenames", "follower_filename"])
SmoothnessScore = namedtuple("SmoothnessScore", ["sparc", "jerk", "log_jerk"])

# TODO: Update this to your path
DATA_FILES_PATH = "C:/Users/Shani/QTM-Wrapper/data_files"

#### QTM configuration ####
load_from_dump = True
skeleton = False
markers_permutation_vector = True
smoothing_factor = 1500
fix_jumps = True

def get_dump_path(file_path):
    return file_path.split(".")[0] + "_interpolated_mirrored_qtm.npz"

def get_follower_from_lf_path(filename):
    # lf filename format: player35_player34_lf_35
    p = re.compile(r'\d+')
    obj0, obj1, leader_obj = p.findall(filename)
    if leader_obj != obj0 and leader_obj != obj1:
        print(f'Bad leader found: {leader_obj} ({obj0, obj1})')
    follower_obj = 0 if leader_obj == obj1 else 1
    return follower_obj

def get_follower_object_id(filename_index, player: PlayerData):
    return player.follower_indexs[filename_index]

def get_follower_marker_id(qtm_obj, filename_index, marker_tag, player: PlayerData):
    return select_marker_by_tag(qtm_obj, player, filename_index, marker_tag)

def get_player_solo_scores(marker, player: PlayerData):
    sparc_scores, jerk_scores, log_jerk_scores = [], [], []
    for filename in player.solo_filenames:
        solo_filepath = os.path.join(DATA_FILES_PATH, filename)
        qtm_obj = QTM(solo_filepath, save_path=get_dump_path(solo_filepath), load=load_from_dump, 
                       num_of_subjects=1, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
        scores = smoothnessMeasurement.get_scores_for_all_segments(qtm_obj, object=0, marker=marker, segment_size=30)
        sparc_scores.extend(scores["sparc"])
        jerk_scores.extend(scores["jerk"])
        log_jerk_scores.extend(scores["log_jerk"])
    return SmoothnessScore(sparc_scores, jerk_scores, log_jerk_scores)

def get_player_follower_score(marker_tag, player: PlayerData):
    sparc_scores, jerk_scores, log_jerk_scores = [], [], []
    for i, filename in enumerate(player.follower_filename):
        follower_filepath = os.path.join(DATA_FILES_PATH, filename)
        qtm_obj = QTM(follower_filepath, save_path=get_dump_path(follower_filepath), load=load_from_dump,
                            num_of_subjects=2, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
        scores = smoothnessMeasurement.get_scores_for_all_segments(qtm_obj, object=get_follower_object_id(i, player), marker=get_follower_marker_id(qtm_obj, i, marker_tag, player), segment_size=30)
        sparc_scores.extend(scores["sparc"])
        jerk_scores.extend(scores["jerk"])
        log_jerk_scores.extend(scores["log_jerk"])
    return SmoothnessScore(sparc_scores, jerk_scores, log_jerk_scores)

'''
def get_scores_of_players_for_markers(players, markers):
    scores_by_markers = defaultdict(dict)
    for marker in markers:
        for player in players:
            solo_scores = get_player_solo_scores(marker, player)
            follower_score = get_player_follower_score(marker, player)
            scores_by_markers[marker] = {player.id: [solo_scores, follower_score]}
        print(f"\n(Marker: {marker}, Player: {player.id}) Solo: {solo_scores}, Follower: {follower_score}")
    return scores_by_markers

def plot_metric_of_solo_vs_lf(scores, metric):
    title_params = dict(fontsize=10, loc="left")
    # TODO: Make generic for all markers?
    scores = scores[0]
    _, ax = plt.subplots()
    for p_id, p_scores in scores.items():
        for solo_score in p_scores[0]:
            ax.scatter(x=p_id, y=getattr(solo_score, metric), label="solo", color="blue")
        ax.scatter(x=p_id, y=getattr(p_scores[1], metric), label="follower", color="green")
    plt.xticks(list(scores.keys()))
    ax.set_title(f"{metric.upper()} score of solo and follower for each player", **title_params)
    ax.set_xlabel("Player ID", fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    # ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
    ax.legend()
    plt.show()
'''

def main():
    marker_to_participant_scores = dict()
    for marker_tag, marker_id in marker_tags_to_solo_marker_id.items():
        marker_to_participant_scores[marker_tag] = dict()
        for player in players_data:
            marker_to_participant_scores[marker_tag][player.id] = dict()
            marker_to_participant_scores[marker_tag][player.id]["solo"] = get_player_solo_scores(marker_id, player)
            marker_to_participant_scores[marker_tag][player.id]["follower"] = get_player_follower_score(marker_tag, player)
            print(f"\n(Marker: {marker_tag}, Player: {player.id}) Solo: {marker_to_participant_scores[marker_tag][player.id]['solo']}, Follower: {marker_to_participant_scores[marker_tag][player.id]['follower']}")

    # scores_by_markers = get_scores_of_players_for_markers(players_data, markers_of_interest)
    # plot_metric_of_solo_vs_lf(scores_by_markers, "sparc")
    # plot_metric_of_solo_vs_lf(scores_by_markers, "jerk")

if __name__ == '__main__':
    main()