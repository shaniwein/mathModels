from qtmWrapper.qtm import *
from qtmWrapper import qtm_types
from qtmWrapper import psd
import matplotlib.pyplot as plt

file_name = f'C:/Users/nbita/PycharmProjects/MovementFeaturesProject/data_files/player33_solo0002.mat'
qtm_obj = QTM(file_name, num_of_subjects=1)
psd_obj = psd.PSDAnalyzer(qtm_obj, num_of_subjects=1)
magnitude_data = np.sqrt(
    np.sum(psd_obj.qtm.get_data(qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY) ** 2, axis=2,
           keepdims=True))
player_ind = 0
solo = psd_obj._analyze_psd(magnitude_data[0:1, ...], psd_obj.qtm.time, psd_obj.qtm.labels[0],
                            qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY, False, True)
for m in range(34):
    plt.bar([i for i in range(39)], solo.psd[0][m][0])
    plt.xticks([i for i in range(39)], np.round(solo.freqs, decimals = 3), fontsize=6)
    plt.xlabel("frequency")
    plt.ylabel("density")
    plt.title(psd_obj.qtm.labels[0][m])
    plt.show()

## Returns
# duet = psd.analyze_duet(qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY)
# solo[0].plot()
# duet.plot()
# plt.show()
# colors = ['b', 'r', 'g', 'y', 'k', 'pink']
# for m in range(102):
#     for freq in range(1, 7):
#         psd_result = solos[0].spectrogram_psd[0, m, :, freq]
#         plt.plot(psd_result, color=colors[freq - 1])
# plt.show()
# for m in range(102):
#     psd_result = solos[0].spectrogram_psd[0, m, :, 20]
#     plt.plot(psd_result)
# plt.title("Density")
# plt.xlabel("time")
# plt.ylabel("density")
# plt.show()
