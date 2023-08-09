from qtmWrapper.qtm import *
from qtmWrapper import qtm_types
from qtmWrapper import psd
import matplotlib.pyplot as plt
import scipy
from collections import namedtuple

#TODO: change path
DATA_FILES_PATH = "C:/Users/nbita/PycharmProjects/MovementFeaturesProject/data_files/"

player_num = 35
total_solo_psd = np.zeros(39)

#TODO: fix selected markers to be by label
selected_markers = [2, 4, 7, 9, 12, 14, 17, 21, 28] #  HeadFront, SpineThoracic12,  LElbowOut, LHand2, RElbowOut, RHand2, WaistBack, LKneeOut, RKneeOut

ks_pval = []
ad_pval = []

for marker in selected_markers:
    ## collect solo data
    for i in range(3):
        player_ind = 0
        file_name = DATA_FILES_PATH +f"player{player_num}_solo000{i+1}.mat"
        qtm_obj = QTM(file_name, num_of_subjects=1)
        psd_obj = psd.PSDAnalyzer(qtm_obj, num_of_subjects=1)
        magnitude_data = np.sqrt(
            np.sum(psd_obj.qtm.get_data(qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY) ** 2, axis=2,
                   keepdims=True))
        solo_psd = psd_obj._analyze_psd(magnitude_data[0:player_ind + 1, ...], psd_obj.qtm.time, psd_obj.qtm.labels[player_ind],
                                    qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY, False, True)
        total_solo_psd = np.add(total_solo_psd, solo_psd.psd[0][marker][0])
    solo_psd_sum = total_solo_psd.sum()
    normalized_solo_psd = total_solo_psd/solo_psd_sum

    ## collect duet data
    duet_names = ["player33_player34_lf_33_1.mat", "player33_player34_lf_34_1.mat", "player33_player34_lf_34_2.mat",
                  "player35_player34_lf_34.mat", "player35_player34_lf_35.mat", "player35_player36_lf_35.mat",
                  "player35_player36_lf_36.mat"]
    player_duets = []
    for name in duet_names:
        if str(player_num) in name and name[21:23] != str(player_num):
            player_duets.append(name)
    total_duet_psd = np.zeros(39)
    for duet in player_duets:
        if duet.index(str(player_num)) == 6:
            player_ind = 0
        else:
            player_ind = 1
        file_name = DATA_FILES_PATH + duet
        qtm_obj = QTM(file_name, num_of_subjects=2)
        psd_obj = psd.PSDAnalyzer(qtm_obj, num_of_subjects=2)
        magnitude_data = np.sqrt(
            np.sum(psd_obj.qtm.get_data(qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY) ** 2, axis=2,
                   keepdims=True))
        duet_psd = psd_obj._analyze_psd(magnitude_data[0:player_ind + 1, ...], psd_obj.qtm.time,
                                        psd_obj.qtm.labels[player_ind],
                                        qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY, False, True)
        total_duet_psd = np.add(total_duet_psd, duet_psd.psd[0][marker][0])
    duet_psd_sum = total_duet_psd.sum()
    print(duet_psd_sum)
    normalized_duet_psd = total_duet_psd / duet_psd_sum

    ## compare solo and LF distributions for given player and marker
    ks_result = scipy.stats.ks_2samp(normalized_solo_psd, normalized_duet_psd)
    ad_result = scipy.stats.anderson_ksamp((normalized_solo_psd,normalized_duet_psd), midrank=True)
    ks_pval.append(ks_result[1])
    ad_pval.append(ad_result[2])

plt.scatter([i for i in range(len(ks_pval))], ks_pval)
plt.show()
plt.scatter([i for i in range(len(ad_pval))], ad_pval)
plt.show()