from qtmWrapper.qtm import *
from qtmWrapper import qtm_types
from qtmWrapper import psd
import matplotlib.pyplot as plt
import scipy
from collections import namedtuple

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

def get_total_psd(total_psd, qtm_obj, selected_markers, num_of_subjects=1, player_ind=0):
    psd_obj = psd.PSDAnalyzer(qtm_obj, num_of_subjects)
    magnitude_data = np.sqrt(
        np.sum(psd_obj.qtm.get_data(qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY) ** 2, axis=2,
               keepdims=True))
    solo_psd = psd_obj._analyze_psd(magnitude_data[0:player_ind + 1, ...], psd_obj.qtm.time,
                                    psd_obj.qtm.labels[player_ind],
                                    qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY, False, True)
    print(selected_markers)
    total_psd = np.add(total_psd, solo_psd.psd[0][selected_markers])
    return total_psd

#TODO: change path
DATA_FILES_PATH = "C:/Users/nbita/PycharmProjects/MovementFeaturesProject/data_files/"
duet_names = ["player33_player34_lf_33_1.mat", "player33_player34_lf_34_1.mat", "player33_player34_lf_34_2.mat",
              "player35_player34_lf_34.mat", "player35_player34_lf_35.mat", "player35_player36_lf_35.mat",
              "player35_player36_lf_36.mat"]
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
solo_selected_markers = [2, 7, 9, 12, 14, 17, 21, 28]

ks_pval = np.empty(8)
ad_pval = np.empty(8)
for player in players_data:
    ## collect solo data
    total_solo_psd = np.zeros((8, 1, 39))
    for solo in range(3):
        player_ind = 0
        file_name = DATA_FILES_PATH +player.solo_filenames[solo]
        qtm_obj = QTM(file_name, num_of_subjects=1)
        total_solo_psd =get_total_psd(total_solo_psd, qtm_obj, solo_selected_markers, 1)
    solo_psd_sum = total_solo_psd.sum(axis=2)
    normalized_solo_psd = np.empty((8, 39))
    for i in range(8):
        normalized_solo_psd[i] = total_solo_psd[i][0]/solo_psd_sum[i]

    ## collect duet data
    total_duet_psd = np.zeros((8, 1, 39))
    for duet in range(len(player.follower_filename)):
        file_name = DATA_FILES_PATH + player.follower_filename[duet]
        qtm_obj = QTM(file_name, num_of_subjects=2)
        duet_selected_markers = select_markers(qtm_obj, player, player.follower_indexs[duet])
        total_duet_psd = get_total_psd(total_duet_psd, qtm_obj, duet_selected_markers, 2, player.follower_indexs[duet])
    duet_psd_sum = total_duet_psd.sum(axis=2)
    normalized_duet_psd = np.empty((8, 39))
    for marker in range(8):
        normalized_duet_psd[marker] = total_duet_psd[marker][0]/duet_psd_sum[marker]

    ## compare solo and LF distributions for given player and marker
        ks_result = scipy.stats.ks_2samp(normalized_solo_psd[marker], normalized_duet_psd[marker])
        ad_result = scipy.stats.anderson_ksamp((normalized_solo_psd[marker],normalized_duet_psd[marker]), midrank=True) ##pvalue floored at 0.001
        ks_pval[marker] = ks_result[1]
        ad_pval[marker] = ad_result[2]
    plt.title(f"kolmogorov-smirnov player {player.id}")
    plt.ylabel("pvalue")
    plt.xlabel("marker")
    plt.xticks([i for i in range(len(ks_pval))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"])
    plt.scatter([i for i in range(len(ks_pval))], ks_pval)
    plt.show()

    plt.title(f"anderson-darling player {player.id}")
    plt.ylabel("pvalue")
    plt.xlabel("marker")
    plt.xticks([i for i in range(len(ad_pval))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"])
    plt.scatter([i for i in range(len(ad_pval))], ad_pval)
    plt.show()