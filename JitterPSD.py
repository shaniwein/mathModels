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
ks_sig = []
ad_sig = []
# for jitter freq
ks_pval_jitter = np.empty(8)
ad_pval_jitter = np.empty(8)
ks_sig_jitter = []
ad_sig_jitter = []

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
    normalized_solo_jitter_psd = np.empty((8, 6))
    for i in range(8):
        normalized_solo_psd[i] = total_solo_psd[i][0]/solo_psd_sum[i]
        normalized_solo_jitter_psd[i] = total_solo_psd[i][0][:6]/solo_psd_sum[i]
    ## collect duet data
    total_duet_psd = np.zeros((8, 1, 39))
    for duet in range(len(player.follower_filename)):
        file_name = DATA_FILES_PATH + player.follower_filename[duet]
        qtm_obj = QTM(file_name, num_of_subjects=2)
        duet_selected_markers = select_markers(qtm_obj, player, player.follower_indexs[duet])
        total_duet_psd = get_total_psd(total_duet_psd, qtm_obj, duet_selected_markers, 2, player.follower_indexs[duet])
    duet_psd_sum = total_duet_psd.sum(axis=2)
    normalized_duet_psd = np.empty((8, 39))
    normalized_duet_jitter_psd = np.empty((8, 6))
    for marker in range(8):
        normalized_duet_psd[marker] = total_duet_psd[marker][0]/duet_psd_sum[marker]
        normalized_duet_jitter_psd[marker] = total_duet_psd[marker][0][:6]/duet_psd_sum[marker]

    ## compare solo and LF distributions for given player and marker
        ks_result = scipy.stats.ks_2samp(normalized_solo_psd[marker], normalized_duet_psd[marker])
        ad_result = scipy.stats.anderson_ksamp((normalized_solo_psd[marker],normalized_duet_psd[marker]), midrank=True) ##pvalue floored at 0.001
        ks_pval[marker] = ks_result[1]
        ad_pval[marker] = ad_result[2]
    ## same for jitter frequencies only
        ks_jitter_result = scipy.stats.ks_2samp(normalized_solo_jitter_psd[marker], normalized_duet_jitter_psd[marker])
        ad_jitter_result = scipy.stats.anderson_ksamp((normalized_solo_jitter_psd[marker], normalized_duet_jitter_psd[marker]),
                                               midrank=True)  ##pvalue floored at 0.001
        ks_pval_jitter[marker] = ks_jitter_result[1]
        ad_pval_jitter[marker] = ad_jitter_result[2]

    ks_sig.append(ks_pval < 0.05)
    ad_sig.append(ad_pval < 0.05)
    ## same for jitter frequencies only
    ks_sig_jitter.append(ks_pval_jitter < 0.05)
    ad_sig_jitter.append(ad_pval_jitter < 0.05)


    ##plot results
    # plt.title(f"kolmogorov-smirnov player {player.id}")
    # plt.ylabel("log pvalue")
    # plt.xlabel("marker")
    # plt.xticks([i for i in range(len(ks_pval))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"], fontsize = 6)
    # plt.scatter([i for i in range(len(ks_pval))], np.log(ks_pval))
    # plt.plot([i for i in range(len(ks_pval))], [np.log(0.05) for i in range(len(ks_pval))], color = 'red')
    # plt.savefig(f"kolmogorov-smirnov player {player.id}.png")
    # plt.show()
    #
    # plt.title(f"anderson-darling player {player.id}")
    # plt.ylabel("log pvalue")
    # plt.xlabel("marker")
    # plt.plot([i for i in range(len(ad_pval))], [np.log(0.05) for i in range(len(ad_pval))], color = 'red')
    # plt.xticks([i for i in range(len(ad_pval))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"], fontsize = 6)
    # plt.scatter([i for i in range(len(ad_pval))], np.log(ad_pval))
    # plt.savefig(f"anderson-darling player {player.id}.png")
    # plt.show()
    ##Jitter freqs
    # plt.title(f"kolmogorov-smirnov player {player.id} Jitter frequencies")
    # plt.ylabel("log pvalue")
    # plt.xlabel("marker")
    # plt.xticks([i for i in range(len(ks_pval_jitter))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"], fontsize = 6)
    # plt.scatter([i for i in range(len(ks_pval_jitter))], np.log(ks_pval_jitter))
    # plt.plot([i for i in range(len(ks_pval_jitter))], [np.log(0.05) for i in range(len(ks_pval_jitter))], color = 'red')
    # plt.savefig(f"kolmogorov-smirnov player - Jitter frequencies {player.id}.png")
    # plt.show()
    #
    # plt.title(f"anderson-darling player {player.id} Jitter frequencies")
    # plt.ylabel("log pvalue")
    # plt.xlabel("marker")
    # plt.plot([i for i in range(len(ad_pval_jitter))], [np.log(0.05) for i in range(len(ad_pval_jitter))], color = 'red')
    # plt.xticks([i for i in range(len(ad_pval_jitter))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"], fontsize = 6)
    # plt.scatter([i for i in range(len(ad_pval_jitter))], np.log(ad_pval_jitter))
    # plt.savefig(f"anderson-darling player - Jitter frequencies {player.id}.png")
    # plt.show()

## plot percentage of players with sig difference btwn LS and solo
plt.bar([i for i in range(len(ks_pval))], np.sum(ks_sig, axis = 0)/len(ks_sig)*100)
plt.title("Percntage of dancers with significant difference (kolmogorov-smirnov)")
plt.xticks([i for i in range(len(ad_pval))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"], fontsize = 6)
plt.ylabel("percentage")
plt.savefig("Percntage of dancers with significant difference (kolmogorov-smirnov).png")
plt.show()
plt.bar([i for i in range(len(ad_pval))], np.sum(ad_sig, axis = 0)/len(ad_sig)*100)
plt.ylabel("percentage")
plt.title("Percntage of dancers with significant difference (anderson-darling)")
plt.xticks([i for i in range(len(ad_pval))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"], fontsize = 6)
plt.savefig("Percntage of dancers with significant difference (anderson-darling).png")
plt.show()

## Jitter frequencies
plt.bar([i for i in range(len(ks_pval_jitter))], np.sum(ks_sig_jitter, axis = 0)/len(ks_sig_jitter)*100)
plt.title("Percntage of dancers with significant difference (kolmogorov-smirnov) - Jitter frequencies")
plt.xticks([i for i in range(len(ad_pval_jitter))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"], fontsize = 6)
plt.ylabel("percentage")
plt.savefig("Percntage of dancers with significant difference - Jitter frequencies (kolmogorov-smirnov).png")
plt.show()
plt.bar([i for i in range(len(ad_pval_jitter))], np.sum(ad_sig_jitter, axis = 0)/len(ad_sig_jitter)*100)
plt.ylabel("percentage")
plt.title("Percntage of dancers with significant difference (anderson-darling) - Jitter frequencies")
plt.xticks([i for i in range(len(ad_pval_jitter))], ["HeadFront", "LElbowOut", "LHand2", "RElbowOut", "RHand2", "WaistBack", "LKneeOut", "RKneeOut"], fontsize = 6)
plt.savefig("Percntage of dancers with significant difference  - Jitter frequencies (anderson-darling).png")
plt.show()