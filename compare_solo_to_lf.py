import os
import re
import json
import numpy as np
import scipy.stats
from collections import namedtuple, defaultdict
from matplotlib import pyplot as plt
from qtmWrapper.qtm import QTM
from common import PlayerData, select_marker_by_tag, players_data, marker_tags_to_solo_marker_id
import smoothnessMeasurement

SmoothnessScore = namedtuple("SmoothnessScore", ["sparc", "jerk", "log_jerk"])

# TODO: Update this to your path
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
    return select_marker_by_tag(qtm_obj, player, player.follower_indexs[filename_index], marker_tag)

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
    for marker_tag, marker_id in marker_tags_to_solo_marker_id.items():
        marker_to_participant_scores[marker_tag] = dict()
        for player in players_data:
            marker_to_participant_scores[marker_tag][player.id] = dict()
            marker_to_participant_scores[marker_tag][player.id]["solo"] = get_player_solo_scores(marker_id, player)
            marker_to_participant_scores[marker_tag][player.id]["follower"] = get_player_follower_score(marker_tag, player)
            print(f"\n(Marker: {marker_tag}, Player: {player.id}) Solo: {marker_to_participant_scores[marker_tag][player.id]['solo']}, Follower: {marker_to_participant_scores[marker_tag][player.id]['follower']}")
            sparc_u, sparc_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["solo"].sparc, marker_to_participant_scores[marker_tag][player.id]["follower"].sparc)
            jerk_u, jerk_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["solo"].jerk, marker_to_participant_scores[marker_tag][player.id]["follower"].jerk)
            log_jerk_u, log_jerk_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["solo"].log_jerk, marker_to_participant_scores[marker_tag][player.id]["follower"].log_jerk)
            marker_to_participant_scores[marker_tag][player.id]["sparc_p"] = sparc_p
            marker_to_participant_scores[marker_tag][player.id]["jerk_p"] = jerk_p
            marker_to_participant_scores[marker_tag][player.id]["log_jerk_p"] = log_jerk_p
            print(f"SPARC: U={sparc_u}, p={sparc_p}")
            print(f"Jerk: U={jerk_u}, p={jerk_p}")
            print(f"Log Jerk: U={log_jerk_u}, p={log_jerk_p}")
    # scores_by_markers = get_scores_of_players_for_markers(players_data, markers_of_interest)
    # plot_metric_of_solo_vs_lf(scores_by_markers, "sparc")
    # plot_metric_of_solo_vs_lf(scores_by_markers, "jerk")
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