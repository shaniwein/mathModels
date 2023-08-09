import os
import re
import numpy as np
from collections import namedtuple, defaultdict
from matplotlib import pyplot as plt
from qtmWrapper.qtm import QTM
import smoothnessMeasurement

PlayerData = namedtuple("PlayerData", ["id", "solo_filenames", "follower_filename"])
SmoothnessScore = namedtuple("SmoothnessScore", ["sparc", "jerk"])

# TODO: Update this to your path
DATA_FILES_PATH = "C:/Users/Shani/QTM-Wrapper/data_files"

# Add players here
players_data = [
    PlayerData(id=35, solo_filenames=["player35_solo0001.mat", "player35_solo0002.mat", "player35_solo0003.mat"], follower_filename="player35_player36_lf_36.mat"),
]

# Add markers here
# TODO: Update to MARKERS instance
markers_of_interest = [
    0,
]

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

def get_player_solo_scores(marker, player: PlayerData):
    scores = []
    for filename in player.solo_filenames:
        solo_filepath = os.path.join(DATA_FILES_PATH, filename)
        qtm_obj = QTM(solo_filepath, save_path=get_dump_path(solo_filepath), load=load_from_dump, 
                       num_of_subjects=1, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
        scores.append(SmoothnessScore(*smoothnessMeasurement.get_mean_scores_for_all_segments(qtm_obj=qtm_obj, object=0, marker=marker)))
    return scores

def get_player_follower_score(marker, player: PlayerData):
    follower_filepath = os.path.join(DATA_FILES_PATH, player.follower_filename)
    qtm_obj = QTM(follower_filepath, save_path=get_dump_path(follower_filepath), load=load_from_dump,
                        num_of_subjects=2, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
    return SmoothnessScore(*smoothnessMeasurement.get_mean_scores_for_all_segments(qtm_obj, object=get_follower_from_lf_path(player.follower_filename), marker=marker))

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

def main():
    scores_by_markers = get_scores_of_players_for_markers(players_data, markers_of_interest)
    plot_metric_of_solo_vs_lf(scores_by_markers, "sparc")
    plot_metric_of_solo_vs_lf(scores_by_markers, "jerk")

if __name__ == '__main__':
    main()