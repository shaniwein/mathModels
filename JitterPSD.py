from qtmWrapper.qtm import *
from qtmWrapper import qtm_types
from qtmWrapper import psd
import matplotlib.pyplot as plt

file_name = "C:/Users/nbita/PycharmProjects/MovementFeaturesProject/data_files/player33_player34_lf_33_1.mat"
qtm_obj = QTM(file_name, num_of_subjects=2)
psd_obj = psd.PSDAnalyzer(qtm_obj, num_of_subjects=2)
magnitude_data = np.sqrt(
    np.sum(psd_obj.qtm.get_data(qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY) ** 2, axis=2,
           keepdims=True))
player_ind = 1
solo = psd_obj._analyze_psd(magnitude_data[0:player_ind+1, ...], psd_obj.qtm.time, psd_obj.qtm.labels[player_ind],
                            qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY, False, True)
#TODO: fix selected markers to be by label
selected_markers = [2, 4, 7, 9, 12, 14, 17, 21, 28] #  HeadFront, SpineThoracic12,  LElbowOut, LHand2, RElbowOut, RHand2, WaistBack, LKneeOut, RKneeOut
for m in selected_markers:
    plt.bar([i for i in range(39)], solo.psd[0][m][0])
    plt.xticks([i for i in range(39)], np.round(solo.freqs, decimals = 3), fontsize=6)
    plt.xlabel("frequency")
    plt.ylabel("density")
    plt.title(psd_obj.qtm.labels[player_ind][m])
    plt.show()