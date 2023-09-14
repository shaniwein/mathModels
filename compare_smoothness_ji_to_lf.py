import os
import re
import json
import numpy as np
import scipy.stats
from collections import namedtuple, defaultdict
from matplotlib import pyplot as plt
from qtmWrapper.qtm import QTM
from common_ji_lf import PlayerData, players_data, select_marker_by_tag, marker_tags_to_ji_marker_id
import smoothnessMeasurement

SmoothnessScore = namedtuple("SmoothnessScore", ["sparc", "jerk", "log_jerk", "velocity_peaks", "mean_speed"])

#### QTM configuration ####
load_from_dump = True
skeleton = False
markers_permutation_vector = True
smoothing_factor = 1500
fix_jumps = True

DATA_FILES_PATH = "C:/Users/Shani/QTM-Wrapper/data_files"
SCORES_JSON_DUMP_PATH = "C:/Users/Shani/QTM-Wrapper/mathModels/scores_data_dump_{}_segment_size.json"

def get_dump_path(file_path):
    return file_path.split(".")[0] + "_interpolated_mirrored_qtm.npz"

# Note: In JI the index of the object is the same as the order in the file name
def get_ji_object_id(filename, player: PlayerData):
    p = re.compile(r'\d+')
    obj0, obj1 = list(map(int, p.findall(filename)))
    if obj0 != player.id and obj1 != player.id:
        raise ValueError(f"player {player.id} not found in filename {filename}")
    return 0 if obj0 == player.id else 1

def get_follower_object_id(filename_index, player: PlayerData):
    return player.follower_indexs[filename_index]

def get_leader_object_id(filename_index, player: PlayerData):
    return player.leader_indexs[filename_index]

def get_follower_marker_id(qtm_obj, filename_index, marker_tag, player: PlayerData):
    return select_marker_by_tag(qtm_obj, player, player.follower_indexs[filename_index], marker_tag)

def get_leader_marker_id(qtm_obj, filename_index, marker_tag, player: PlayerData):
    return select_marker_by_tag(qtm_obj, player, player.leader_indexs[filename_index], marker_tag)

def get_player_ji_scores(marker_id, player: PlayerData, segment_size):
    sparc_scores, jerk_scores, log_jerk_scores, velocity_peaks, mean_speed = [], [], [], [], []
    for filename in player.ji_filenames:
        ji_filepath = os.path.join(DATA_FILES_PATH, filename)
        qtm_obj = QTM(ji_filepath, save_path=get_dump_path(ji_filepath), load=load_from_dump, 
                       num_of_subjects=2, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
        scores = smoothnessMeasurement.get_scores_for_segments_by_segment_size(qtm_obj, object=get_ji_object_id(filename, player), marker=marker_id, segment_size=segment_size)
        sparc_scores.extend(scores["sparc"])
        jerk_scores.extend(scores["jerk"])
        log_jerk_scores.extend(scores["log_jerk"])
        velocity_peaks.extend(scores["velocity_peaks"])
        mean_speed.extend(scores["mean_speed"])
    return SmoothnessScore(sparc_scores, jerk_scores, log_jerk_scores, velocity_peaks, mean_speed)

def get_player_follower_scores(marker_tag, player: PlayerData, segment_size):
    sparc_scores, jerk_scores, log_jerk_scores, velocity_peaks, mean_speed = [], [], [], [], []
    for i, filename in enumerate(player.follower_filenames):
        follower_filepath = os.path.join(DATA_FILES_PATH, filename)
        qtm_obj = QTM(follower_filepath, save_path=get_dump_path(follower_filepath), load=load_from_dump,
                            num_of_subjects=2, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
        scores = smoothnessMeasurement.get_scores_for_segments_by_segment_size(qtm_obj, object=get_follower_object_id(i, player), marker=get_follower_marker_id(qtm_obj, i, marker_tag, player), segment_size=segment_size)
        sparc_scores.extend(scores["sparc"])
        jerk_scores.extend(scores["jerk"])
        log_jerk_scores.extend(scores["log_jerk"])
        velocity_peaks.extend(scores["velocity_peaks"])
        mean_speed.extend(scores["mean_speed"])
    return SmoothnessScore(sparc_scores, jerk_scores, log_jerk_scores, velocity_peaks, mean_speed)

def get_player_leader_scores(marker_tag, player: PlayerData, segment_size):
    sparc_scores, jerk_scores, log_jerk_scores, velocity_peaks, mean_speed = [], [], [], [], []
    for i, filename in enumerate(player.leader_filenames):
        leader_filepath = os.path.join(DATA_FILES_PATH, filename)
        qtm_obj = QTM(leader_filepath, save_path=get_dump_path(leader_filepath), load=load_from_dump,
                            num_of_subjects=2, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
        scores = smoothnessMeasurement.get_scores_for_segments_by_segment_size(qtm_obj, object=get_leader_object_id(i, player), marker=get_leader_marker_id(qtm_obj, i, marker_tag, player), segment_size=segment_size)
        sparc_scores.extend(scores["sparc"])
        jerk_scores.extend(scores["jerk"])
        log_jerk_scores.extend(scores["log_jerk"])
        velocity_peaks.extend(scores["velocity_peaks"])
        mean_speed.extend(scores["mean_speed"])
    return SmoothnessScore(sparc_scores, jerk_scores, log_jerk_scores, velocity_peaks, mean_speed)

def get_sig_diff_percentage(values_array):
    values_array = np.array(values_array)
    return np.sum(values_array < 0.05) / len(values_array) * 100

def get_percentage_of_significant_diff_per_marker_for_each_metric(scores):
    # Output format: {marker: percentage}
    sparc_percentages, jerk_percentages, log_jerk_percentages, velocity_peaks_percentages, mean_speed_percentages = {}, {}, {}, {}, {} 
    for marker, player_dicts in scores.items():
        sparc_p_values, jerk_p_values, log_jerk_p_values, velocity_peaks_p_values, mean_speed_p_values = [], [], [], [], []
        for player_scores in player_dicts.values():
            sparc_p_values.append(player_scores["sparc_p"])
            jerk_p_values.append(player_scores["jerk_p"])
            log_jerk_p_values.append(player_scores["log_jerk_p"])
            velocity_peaks_p_values.append(player_scores["velocity_peaks_p"])
            mean_speed_p_values.append(player_scores["mean_speed_p"])
        sparc_percentages[marker] = get_sig_diff_percentage(sparc_p_values)
        jerk_percentages[marker] = get_sig_diff_percentage(jerk_p_values)
        log_jerk_percentages[marker] = get_sig_diff_percentage(log_jerk_p_values)
        velocity_peaks_percentages[marker] = get_sig_diff_percentage(velocity_peaks_p_values)
        mean_speed_percentages[marker] = get_sig_diff_percentage(mean_speed_p_values)
    return sparc_percentages, jerk_percentages, log_jerk_percentages, velocity_peaks_percentages, mean_speed_percentages

def plot_percentage_bar_chart(ax, percentages, scores, label):
    ax.bar(x=list(scores.keys()), height=percentages, width=0.3, alpha=0.7, color="blue")
    # ax.set_title(f"Percentage of significant differences in {label} smoothness score for each marker", fontsize=10, loc="left")
    ax.set_title(f"{label}", fontsize=10, loc="left")
    ax.set_xlabel("Marker", fontsize=8)
    ax.set_ylabel("Percentage", fontsize=8)
    ax.tick_params(axis="x", labelsize=7, rotation=40)
    ax.set_ylim(0, 100)

def plot_percentage_scatter_chart(ax, segment_size_to_marker_to_percentage, label):
    # Data format: {segment_size: {marker: percentage}}
    # The x axis is the marker, the y axis is the percentage, and the color is the segment size
    for segment_size, marker_to_percentage in segment_size_to_marker_to_percentage.items():
        for marker, percentage in marker_to_percentage.items():
            ax.scatter(marker, percentage, color=f"C{segment_size}", alpha=0.7)
    ax.set_title(f"{label}", fontsize=10, loc="left")
    ax.set_xlabel("Marker", fontsize=8)
    ax.set_ylabel("Percentage", fontsize=8)
    ax.legend(title="Segment size", loc='upper right', bbox_to_anchor=(1.5, 1))
    ax.tick_params(axis="x", labelsize=7, rotation=40)
    ax.set_ylim(0, 100)
      
def scatter_sig_diff_percentages_per_marker_and_segment_size_for_each_marker(segment_size_to_list_of_percentages):
# def scatter_sig_diff_percentages_per_marker_and_segment_size_for_each_marker(sparc_percentages, jerk_percentages, log_jerk_percentages, velocity_peaks_percentages, mean_speed_percentages):
    # Data format: {segment_size: [{marker: sparc_percentage}, ... , {marker: dlj_percentage}, ...]}
    sparc_percentages, jerk_percentages, log_jerk_percentages, velocity_peaks_percentages, mean_speed_percentages = [], [], [], [], []
    for segment_size, list_of_percentages in segment_size_to_list_of_percentages.items():
        sparc_percentages.append({segment_size: list_of_percentages[0]})
        jerk_percentages.append({segment_size: list_of_percentages[1]})
        log_jerk_percentages.append({segment_size: list_of_percentages[2]})
        velocity_peaks_percentages.append({segment_size: list_of_percentages[3]})
        mean_speed_percentages.append({segment_size: list_of_percentages[4]})
    _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    plot_percentage_scatter_chart(ax1, sparc_percentages, "SPARC")
    plot_percentage_scatter_chart(ax2, jerk_percentages, "DLJ")
    plot_percentage_scatter_chart(ax3, log_jerk_percentages, "LDLJ")
    plot_percentage_scatter_chart(ax4, velocity_peaks_percentages, "VP")
    plot_percentage_scatter_chart(ax5, mean_speed_percentages, "MS")
    plt.suptitle("Percentage of dancers with significant differences in smoothness scores", fontsize=12)
    plt.tight_layout()
    plt.show()
   
# def plot_the_percentage_of_significant_differences_per_marker(scores):
#     sparc_percentages, jerk_percentages, log_jerk_percentages = [], [], []
#     for marker, player_dicts in scores.items():
#         sparc_p_values, jerk_p_values, log_jerk_p_values, velocity_peaks_p_values, mean_speed_p_values = [], [], [], [], []
#         for player, player_scores in player_dicts.items():
#             sparc_p_values.append(player_scores["sparc_p"])
#             jerk_p_values.append(player_scores["jerk_p"])
#             log_jerk_p_values.append(player_scores["log_jerk_p"])
#             velocity_peaks_p_values.append(player_scores["velocity_peaks_p"])
#             mean_speed_p_values.append(player_scores["mean_speed_p"])
#         sparc_p_values = np.array(sparc_p_values)
#         jerk_p_values = np.array(jerk_p_values)
#         log_jerk_p_values = np.array(log_jerk_p_values)
#         sparc_percentages.append(np.sum(sparc_p_values < 0.05) / len(sparc_p_values) * 100)
#         jerk_percentages.append(np.sum(jerk_p_values < 0.05) / len(jerk_p_values) * 100)
#         log_jerk_percentages.append(np.sum(log_jerk_p_values < 0.05) / len(log_jerk_p_values) * 100)
#     _, (ax1, ax2) = plt.subplots(1, 2)
#     plot_percentage_bar_chart(ax1, sparc_percentages, scores, "SPARC Metric")
#     plot_percentage_bar_chart(ax2, jerk_percentages, scores, "Dimenssionless Jerk Metric")
#     plt.suptitle("Percentage of dancers with significant differences in smoothness scores", fontsize=12)
#     plt.tight_layout()
#     plt.show()

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

# """
# Find differences between the smoothness of the follower and leader in the LF task for each marker and for each measure.
# This is similar to the analysis done in the paper "The mirror game as a paradigm for studying the dynamics of two people improvising motion together"
# """
# def plot_the_percentage_of_significant_differences_in_leader_and_follower_per_marker(scores):
#     sparc_percentages, jerk_percentages, log_jerk_percentages = [], [], []
#     for marker, player_dicts in scores.items():
#         sparc_p_values = []
#         jerk_p_values = []
#         log_jerk_p_values = []
#         for _, player_scores in player_dicts.items():
#             sparc_p_values.append(player_scores["leader_follower_sparc_p"])
#             jerk_p_values.append(player_scores["leader_follower_jerk_p"])
#             log_jerk_p_values.append(player_scores["leader_follower_log_jerk_p"])
#         sparc_p_values = np.array(sparc_p_values)
#         jerk_p_values = np.array(jerk_p_values)
#         log_jerk_p_values = np.array(log_jerk_p_values)
#         sparc_percentages.append(np.sum(sparc_p_values < 0.05) / len(sparc_p_values) * 100)
#         jerk_percentages.append(np.sum(jerk_p_values < 0.05) / len(jerk_p_values) * 100)
#         log_jerk_percentages.append(np.sum(log_jerk_p_values < 0.05) / len(log_jerk_p_values) * 100)
#     _, (ax1, ax2) = plt.subplots(1, 2)
#     plot_percentage_bar_chart(ax1, sparc_percentages, scores, "SPARC Metric")
#     plot_percentage_bar_chart(ax2, jerk_percentages, scores, "Dimenssionless Jerk Metric")
#     plt.suptitle("Percentage of dancers with significant differences between their roles as leaderer and as follower", fontsize=12)
#     plt.tight_layout()
#     plt.show()

# def plot_p_values_per_player_leader_follower(scores):
#     fig, axes = plt.subplots(1, 4, sharey=True, figsize=(15, 5))
#     fig.suptitle(f"p-values of smoothness scores for each player", fontsize=12)
#     players_sparc_p_val_per_marker = defaultdict(dict)
#     players_jerk_p_val_per_marker = defaultdict(dict)
#     players_log_jerk_p_val_per_marker = defaultdict(dict)
#     for marker, player_dicts in scores.items():
#         for player, player_scores in player_dicts.items():
#             players_sparc_p_val_per_marker[player][marker] = player_scores["leader_follower_sparc_p"]
#             players_jerk_p_val_per_marker[player][marker] = player_scores["leader_follower_jerk_p"]
#             players_log_jerk_p_val_per_marker[player][marker] = player_scores["leader_follower_log_jerk_p"]
#     for i, (player, markers_to_sparc_p_vals) in enumerate(players_sparc_p_val_per_marker.items()):
#         axes[i].set_title(f"player {player} (leader-follower)")
#         axes[i].set_ylabel("log pvalue")
#         axes[i].set_xlabel("marker")
#         axes[i].scatter(list(markers_to_sparc_p_vals.keys()), np.log(list(markers_to_sparc_p_vals.values())), label="SPARC", color="blue", alpha=0.7)
#         axes[i].tick_params(axis="x", labelsize=7, rotation=40)
#     for i, (player, markers_to_jerk_p_vals) in enumerate(players_jerk_p_val_per_marker.items()):
#         axes[i].scatter(list(markers_to_jerk_p_vals.keys()), np.log(list(markers_to_jerk_p_vals.values())), label="Jerk", color="green", alpha=0.7)
#     for ax in axes:
#         ax.axhline(y=np.log(0.05), color='red', label='log(0.05)')
#     axes[-1].legend(loc='upper right', bbox_to_anchor=(2, 1)) 
#     plt.tight_layout()
#     plt.show()

def get_data_metric_scores(segment_size):
    marker_to_participant_scores = dict()
    for marker_tag, marker_id in marker_tags_to_ji_marker_id.items():
        marker_to_participant_scores[marker_tag] = dict()
        for player in players_data:
            marker_to_participant_scores[marker_tag][player.id] = dict()
            marker_to_participant_scores[marker_tag][player.id]["ji"] = get_player_ji_scores(marker_id, player, segment_size)
            marker_to_participant_scores[marker_tag][player.id]["follower"] = get_player_follower_scores(marker_tag, player, segment_size)
            marker_to_participant_scores[marker_tag][player.id]["leader"] = get_player_leader_scores(marker_tag, player, segment_size)
    return marker_to_participant_scores

# def get_data_scores():
#     marker_to_participant_scores = dict()
#     for marker_tag, marker_id in marker_tags_to_ji_marker_id.items():
#         marker_to_participant_scores[marker_tag] = dict()
#         for player in players_data:
#             marker_to_participant_scores[marker_tag][player.id] = dict()
#             marker_to_participant_scores[marker_tag][player.id]["ji"] = get_player_ji_scores(marker_id, player)
#             marker_to_participant_scores[marker_tag][player.id]["follower"] = get_player_follower_scores(marker_tag, player)
#             marker_to_participant_scores[marker_tag][player.id]["leader"] = get_player_leader_scores(marker_tag, player)
#             sparc_u, sparc_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["ji"].sparc, marker_to_participant_scores[marker_tag][player.id]["follower"].sparc)
#             jerk_u, jerk_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["ji"].jerk, marker_to_participant_scores[marker_tag][player.id]["follower"].jerk)
#             log_jerk_u, log_jerk_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["ji"].log_jerk, marker_to_participant_scores[marker_tag][player.id]["follower"].log_jerk)
#             marker_to_participant_scores[marker_tag][player.id]["sparc_p"] = sparc_p
#             marker_to_participant_scores[marker_tag][player.id]["jerk_p"] = jerk_p
#             marker_to_participant_scores[marker_tag][player.id]["log_jerk_p"] = log_jerk_p
#             _, leader_follower_sparc_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["leader"].sparc, marker_to_participant_scores[marker_tag][player.id]["follower"].sparc)
#             _, leader_follower_jerk_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["leader"].jerk, marker_to_participant_scores[marker_tag][player.id]["follower"].jerk)
#             _, leader_follower_log_jerk_p = scipy.stats.mannwhitneyu(marker_to_participant_scores[marker_tag][player.id]["leader"].log_jerk, marker_to_participant_scores[marker_tag][player.id]["follower"].log_jerk)
#             marker_to_participant_scores[marker_tag][player.id]["leader_follower_sparc_p"] = leader_follower_sparc_p
#             marker_to_participant_scores[marker_tag][player.id]["leader_follower_jerk_p"] = leader_follower_jerk_p
#             marker_to_participant_scores[marker_tag][player.id]["leader_follower_log_jerk_p"] = leader_follower_log_jerk_p
#     return marker_to_participant_scores

def mann_whitney_by_marker_and_player_for_each_metric(scores, group1, group2):
    marker_to_participant_scores = dict()
    for marker_tag in marker_tags_to_ji_marker_id.keys():
        marker_to_participant_scores[marker_tag] = dict()
        for player in players_data:
            sparc_u, sparc_p = scipy.stats.mannwhitneyu(scores[marker_tag][player.id][group1].sparc, scores[marker_tag][player.id][group2].sparc)
            jerk_u, jerk_p = scipy.stats.mannwhitneyu(scores[marker_tag][player.id][group1].jerk, scores[marker_tag][player.id][group2].jerk)
            log_jerk_u, log_jerk_p = scipy.stats.mannwhitneyu(scores[marker_tag][player.id][group1].log_jerk, scores[marker_tag][player.id][group2].log_jerk)
            velocity_peaks_u, velocity_peaks_p = scipy.stats.mannwhitneyu(scores[marker_tag][player.id][group1].velocity_peaks, scores[marker_tag][player.id][group2].velocity_peaks)
            mean_speed_u, mean_speed_p = scipy.stats.mannwhitneyu(scores[marker_tag][player.id][group1].mean_speed, scores[marker_tag][player.id][group2].mean_speed)
            marker_to_participant_scores[marker_tag][player.id]["sparc_p"] = sparc_p
            marker_to_participant_scores[marker_tag][player.id]["jerk_p"] = jerk_p
            marker_to_participant_scores[marker_tag][player.id]["log_jerk_p"] = log_jerk_p
            marker_to_participant_scores[marker_tag][player.id]["velocity_peaks_p"] = velocity_peaks_p
            marker_to_participant_scores[marker_tag][player.id]["mean_speed_p"] = mean_speed_p
    return marker_to_participant_scores

def get_full_data_scores():
    segment_size_to_marker_to_participant_scores = dict()
    for segment_size in [3, 5, 10, 30]:
        score_dump_path = SCORES_JSON_DUMP_PATH.format(segment_size)
        if not os.path.exists(score_dump_path):
            marker_to_participant_scores = get_data_metric_scores(segment_size)
            segment_size_to_marker_to_participant_scores[segment_size] = marker_to_participant_scores
            with open(score_dump_path, "w") as outfile:
                json.dump(marker_to_participant_scores, outfile)
        else:
            segment_size_to_marker_to_participant_scores[segment_size] = json.load(open(score_dump_path))
    return segment_size_to_marker_to_participant_scores

def main():
    segment_size_to_marker_to_participant_scores = get_full_data_scores()
    ji_follower_comparison_p_vals = dict()
    leader_follower_comparison_p_vals = dict()
    ji_follower_percentages_by_segment_size = list()
    leader_follower_percentages_by_segment_size = list()
    for segment_size, marker_to_participant_scores in segment_size_to_marker_to_participant_scores.items():
        # JI vs follower comparison
        ji_follower_comparison_p_vals[segment_size] = mann_whitney_by_marker_and_player_for_each_metric(marker_to_participant_scores, group1="ji", group2="follower")
        ji_follower_percentages_by_segment_size[segment_size]= get_percentage_of_significant_diff_per_marker_for_each_metric(ji_follower_comparison_p_vals[segment_size])
        # Leader vs follower comparison
        leader_follower_comparison_p_vals[segment_size] = mann_whitney_by_marker_and_player_for_each_metric(marker_to_participant_scores, group1="leader", group2="follower")
        leader_follower_percentages_by_segment_size[segment_size] = get_percentage_of_significant_diff_per_marker_for_each_metric(leader_follower_comparison_p_vals[segment_size])
    scatter_sig_diff_percentages_per_marker_and_segment_size_for_each_marker(ji_follower_percentages_by_segment_size) 
    scatter_sig_diff_percentages_per_marker_and_segment_size_for_each_marker(leader_follower_percentages_by_segment_size)
    # plot_the_percentage_of_significant_differences_per_marker(marker_to_participant_scores)
    # plot_p_values_per_player(marker_to_participant_scores)
    # compare_leader_follower_smoothness_scores(marker_to_participant_scores)

if __name__ == '__main__':
   main()