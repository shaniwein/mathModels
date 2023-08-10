from qtmWrapper.qtm import *
from qtmWrapper import qtm_types
from qtmWrapper import psd
import matplotlib.pyplot as plt
import scipy
from collections import namedtuple

def select_markers(qtm_obj, player_ind):
    """
    select only chosen markers from duet: HeadFront, SpineThoracic12,  LElbowOut, LHand2, RElbowOut, RHand2, WaistBack, LKneeOut, RKneeOut

    :return: indexs of selected markers
    """
    #TODO: figure out if player data is in the correct order
    marker_labels = [f"player{player_num}_HeadFront", f"player{player_num}_SpineThoracic12",
                     f"player{player_num}_LElbowOut", f"player{player_num}_LHand2", f"player{player_num}_RElbowOut",
                     f"player{player_num}_RHand2", f"player{player_num}_WaistBack", f"player{player_num}_LKneeOut",
                     f"player{player_num}_RKneeOut"]
    selected_markers = []
    for i, label in np.ndenumerate(qtm_obj.labels[player_ind]):
        print(label)
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
    total_psd = np.add(total_psd, solo_psd.psd[0][selected_markers])
    return total_psd

#TODO: change path
DATA_FILES_PATH = "C:/Users/nbita/PycharmProjects/MovementFeaturesProject/data_files/"
duet_names = ["player33_player34_lf_33_1.mat", "player33_player34_lf_34_1.mat", "player33_player34_lf_34_2.mat",
              "player35_player34_lf_34.mat", "player35_player34_lf_35.mat", "player35_player36_lf_35.mat",
              "player35_player36_lf_36.mat"]
player_numbers = [33, 34, 35, 36]
#TODO: run for all players
player_num = 35
ks_pval = []
ad_pval = []

## collect solo data
solo_selected_markers = [2, 4, 7, 9, 12, 14, 17, 21, 28]
total_solo_psd = np.zeros((9, 1, 39))
for i in range(3):
    player_ind = 0
    file_name = DATA_FILES_PATH +f"player{player_num}_solo000{i+1}.mat"
    qtm_obj = QTM(file_name, num_of_subjects=1)
    total_solo_psd =get_total_psd(total_solo_psd, qtm_obj, solo_selected_markers, 1)
solo_psd_sum = total_solo_psd.sum(axis=2)
normalized_solo_psd = np.empty((9, 39))
for i in range(9):
    normalized_solo_psd[i] = total_solo_psd[i][0]/solo_psd_sum[i]

## collect duet data
player_duets = []
for name in duet_names:
    if str(player_num) in name and name[21:23] != str(player_num):
        player_duets.append(name)

total_duet_psd = np.zeros((9, 1, 39))
for duet in player_duets:
    if duet.index(str(player_num)) == 6:
        player_ind = 0
    else:
        player_ind = 1
    file_name = DATA_FILES_PATH + duet
    qtm_obj = QTM(file_name, num_of_subjects=2)
    duet_selected_markers = select_markers(qtm_obj, player_ind)
    total_duet_psd = get_total_psd(total_duet_psd, qtm_obj, duet_selected_markers, 2, player_ind)
duet_psd_sum = total_duet_psd.sum(axis=2)
normalized_duet_psd = np.empty((9, 39))
for i in range(9):
    normalized_duet_psd[i] = total_duet_psd[i][0]/duet_psd_sum[i]

## compare solo and LF distributions for given player and marker
marker = 0
ks_result = scipy.stats.ks_2samp(normalized_solo_psd[marker], normalized_duet_psd[marker])
ad_result = scipy.stats.anderson_ksamp((normalized_solo_psd[marker],normalized_duet_psd[marker]), midrank=True) ##pvalue floored at 0.001
ks_pval.append(ks_result[1])
ad_pval.append(ad_result[2])

plt.scatter([i for i in range(len(ks_pval))], ks_pval)
plt.show()
plt.scatter([i for i in range(len(ad_pval))], ad_pval)
plt.show()