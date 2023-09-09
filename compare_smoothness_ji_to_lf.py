import os
import re
import json
import numpy as np
import scipy.stats
from collections import namedtuple, defaultdict
from matplotlib import pyplot as plt
from qtmWrapper.qtm import QTM
from common_ji_lf import PlayerData, players_data, select_marker_by_tag, marker_tags
import smoothnessMeasurement

SmoothnessScore = namedtuple("SmoothnessScore", ["sparc", "jerk", "log_jerk"])

DATA_FILES_PATH = "C:/Users/Shani/QTM-Wrapper/data_files"
SCORES_JSON_DUMP_PATH = "C:/Users/Shani/QTM-Wrapper/mathModels/scores_data_dump.json"

#### QTM configuration ####
load_from_dump = True
skeleton = False
markers_permutation_vector = True
smoothing_factor = 1500
fix_jumps = True

def get_dump_path(file_path):
    return file_path.split(".")[0] + "_interpolated_mirrored_qtm.npz"

# TODO: Fix!
def get_ji_object_id(filename, player: PlayerData):
# def get_ji_object_id(filename_index, player: PlayerData):
    # return player.ji_indexes[filename_index]
    p = re.compile(r'\d+')
    obj0, obj1 = list(map(int, p.findall(filename)))
    print(f"obj0={obj0}, obj1={obj1}")
    if obj0 != player.id and obj1 != player.id:
        raise ValueError(f"player {player.id} not found in filename {filename}")
    return 0 if obj0 == player.id else 1

def get_ji_marker_id(qtm_obj, marker_tag, object_id, player: PlayerData):
# def get_ji_marker_id(qtm_obj, filename_index, marker_tag, player: PlayerData):
    return select_marker_by_tag(qtm_obj, player, object_id, marker_tag)
    # return select_marker_by_tag(qtm_obj, player, player.ji_indexs[filename_index], marker_tag)

############

def get_follower_object_id(filename_index, player: PlayerData):
    return player.follower_indexs[filename_index]

def get_follower_marker_id(qtm_obj, filename_index, marker_tag, player: PlayerData):
    return select_marker_by_tag(qtm_obj, player, player.follower_indexs[filename_index], marker_tag)

def get_player_ji_scores(marker_tag, player: PlayerData):
    sparc_scores, jerk_scores, log_jerk_scores = [], [], []
    for i, filename in enumerate(player.ji_filenames):
        ji_filepath = os.path.join(DATA_FILES_PATH, filename)
        qtm_obj = QTM(ji_filepath, save_path=get_dump_path(ji_filepath), load=load_from_dump, 
                       num_of_subjects=2, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
        # TODO: Find the right object num (and put in players data like follower indexes)
        # scores = smoothnessMeasurement.get_scores_for_all_segments(qtm_obj, object=get_ji_object_id(i, player), marker=get_ji_marker_id(qtm_obj, i, marker_tag, player), segment_size=30)
        obj_id = get_ji_object_id(filename, player)
        scores = smoothnessMeasurement.get_scores_for_all_segments(qtm_obj, object=obj_id, marker=get_ji_marker_id(qtm_obj, marker_tag, obj_id, player), segment_size=30)
        sparc_scores.extend(scores["sparc"])
        jerk_scores.extend(scores["jerk"])
        log_jerk_scores.extend(scores["log_jerk"])
    return SmoothnessScore(sparc_scores, jerk_scores, log_jerk_scores)

def get_player_follower_score(marker_tag, player: PlayerData):
    sparc_scores, jerk_scores, log_jerk_scores = [], [], []
    for i, filename in enumerate(player.follower_filenames):
        follower_filepath = os.path.join(DATA_FILES_PATH, filename)
        qtm_obj = QTM(follower_filepath, save_path=get_dump_path(follower_filepath), load=load_from_dump,
                            num_of_subjects=2, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
        scores = smoothnessMeasurement.get_scores_for_all_segments(qtm_obj, object=get_follower_object_id(i, player), marker=get_follower_marker_id(qtm_obj, i, marker_tag, player), segment_size=30)
        sparc_scores.extend(scores["sparc"])
        jerk_scores.extend(scores["jerk"])
        log_jerk_scores.extend(scores["log_jerk"])
    return SmoothnessScore(sparc_scores, jerk_scores, log_jerk_scores)

def plot_percentage_bar_chart(ax, percentages, scores, label):
    ax.bar(x=list(scores.keys()), height=percentages, width=0.3, alpha=0.7, color="blue")
    # ax.set_title(f"Percentage of significant differences in {label} smoothness score for each marker", fontsize=10, loc="left")
    ax.set_title(f"{label}", fontsize=10, loc="left")
    ax.set_xlabel("Marker", fontsize=8)
    ax.set_ylabel("Percentage", fontsize=8)
    ax.tick_params(axis="x", labelsize=7, rotation=40)
    ax.set_ylim(0, 100)

def plot_the_percentage_of_significant_differences_per_marker(scores):
    sparc_percentages, jerk_percentages, log_jerk_percentages = [], [], []
    for marker, player_dicts in scores.items():
        sparc_p_values = []
        jerk_p_values = []
        log_jerk_p_values = []
        for player, player_scores in player_dicts.items():
            sparc_p_values.append(player_scores["sparc_p"])
            jerk_p_values.append(player_scores["jerk_p"])
            log_jerk_p_values.append(player_scores["log_jerk_p"])
        sparc_p_values = np.array(sparc_p_values)
        jerk_p_values = np.array(jerk_p_values)
        log_jerk_p_values = np.array(log_jerk_p_values)
        sparc_percentages.append(np.sum(sparc_p_values < 0.05) / len(sparc_p_values) * 100)
        jerk_percentages.append(np.sum(jerk_p_values < 0.05) / len(jerk_p_values) * 100)
        log_jerk_percentages.append(np.sum(log_jerk_p_values < 0.05) / len(log_jerk_p_values) * 100)
    _, (ax1, ax2) = plt.subplots(1, 2)
    plot_percentage_bar_chart(ax1, sparc_percentages, scores, "SPARC Metric")
    plot_percentage_bar_chart(ax2, jerk_percentages, scores, "Dimenssionless Jerk Metric")
    plt.suptitle("Percentage of dancers with significant differences in smoothness scores", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_p_values_per_player(scores):
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(15, 5))
    fig.suptitle(f"p-values of smoothness scores for each player", fontsize=12)
    players_sparc_p_val_per_marker = defaultdict(dict)
    players_jerk_p_val_per_marker = defaultdict(dict)
    players_log_jerk_p_val_per_marker = defaultdict(dict)
    for marker, player_dicts in scores.items():
        for player, player_scores in player_dicts.items():
            players_sparc_p_val_per_marker[player][marker] = player_scores["sparc_p"]
            players_jerk_p_val_per_marker[player][marker] = player_scores["jerk_p"]
            players_log_jerk_p_val_per_marker[player][marker] = player_scores["log_jerk_p"]
    for i, (player, markers_to_sparc_p_vals) in enumerate(players_sparc_p_val_per_marker.items()):
        axes[i].set_title(f"player {player}")
        axes[i].set_ylabel("log pvalue")
        axes[i].set_xlabel("marker")
        axes[i].scatter(list(markers_to_sparc_p_vals.keys()), np.log(list(markers_to_sparc_p_vals.values())), label="SPARC", color="blue", alpha=0.7)
        axes[i].tick_params(axis="x", labelsize=7, rotation=40)
    for i, (player, markers_to_jerk_p_vals) in enumerate(players_jerk_p_val_per_marker.items()):
        axes[i].scatter(list(markers_to_jerk_p_vals.keys()), np.log(list(markers_to_jerk_p_vals.values())), label="Jerk", color="green", alpha=0.7)
    for ax in axes:
        ax.axhline(y=np.log(0.05), color='red', label='log(0.05)')
    axes[-1].legend(loc='upper right', bbox_to_anchor=(2, 1)) 
    plt.tight_layout()
    plt.show()

def get_data_scores():
    marker_to_participant_scores = dict()
    for marker_tag in marker_tags:
        marker_to_participant_scores[marker_tag] = dict()
        for player in players_data:
            marker_to_participant_scores[marker_tag][player.id] = dict()
            marker_to_participant_scores[marker_tag][player.id]["ji"] = get_player_ji_scores(marker_tag, player)
            marker_to_participant_scores[marker_tag][player.id]["follower"] = get_player_follower_score(marker_tag, player)
            sparc_u, sparc_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["ji"].sparc, marker_to_participant_scores[marker_tag][player.id]["follower"].sparc)
            jerk_u, jerk_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["ji"].jerk, marker_to_participant_scores[marker_tag][player.id]["follower"].jerk)
            log_jerk_u, log_jerk_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["ji"].log_jerk, marker_to_participant_scores[marker_tag][player.id]["follower"].log_jerk)
            marker_to_participant_scores[marker_tag][player.id]["sparc_p"] = sparc_p
            marker_to_participant_scores[marker_tag][player.id]["jerk_p"] = jerk_p
            marker_to_participant_scores[marker_tag][player.id]["log_jerk_p"] = log_jerk_p
            print(f"SPARC: U={sparc_u}, p={sparc_p}")
            print(f"Jerk: U={jerk_u}, p={jerk_p}")
            print(f"Log Jerk: U={log_jerk_u}, p={log_jerk_p}")
    return marker_to_participant_scores

def main():
    if not os.path.exists(SCORES_JSON_DUMP_PATH):
        marker_to_participant_scores = get_data_scores()
        with open("w") as outfile:
            json.dump(marker_to_participant_scores, outfile)
    else:
        marker_to_participant_scores = json.load(open(SCORES_JSON_DUMP_PATH))
    print(f"{marker_to_participant_scores}")
    plot_the_percentage_of_significant_differences_per_marker(marker_to_participant_scores)
    plot_p_values_per_player(marker_to_participant_scores)

if __name__ == '__main__':
    main()