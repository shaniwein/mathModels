from qtmWrapper.qtm import *
from qtmWrapper import qtm_types
from qtmWrapper import psd
import matplotlib.pyplot as plt

file_name = f'player33_solo0002.mat'
qtm_obj = QTM(file_name, num_of_subjects=1)
psd = psd.PSDAnalyzer(qtm_obj, num_of_subjects=1)
solos = psd.analyze_solos(qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY)

#
# for marker in range(34):
#     x_acc = solos[0].spectrogram_psd[0][marker]
#     y_acc = qtm_obj.accelerations[player_ind][marker][1]
#     z_acc = qtm_obj.accelerations[player_ind][marker][2]
#     total_acc = np.sqrt(pow(x_acc, 2) + pow(y_acc, 2) + pow(z_acc, 2))
#     accelerations.append(total_acc)

# duet = psd.analyze_duet(qtm_types.DataType.MARKERS, qtm_types.AnalysisType.VELOCITY)
solos[0].plot()
# duet.plot()
plt.show()
colors = ['b' ,'r', 'g', 'y', 'k', 'pink']
for m in range(102):
    for freq in range(1,7):
        psd_result = solos[0].spectrogram_psd[0, m, :, freq]
        plt.plot(psd_result, color = colors[freq-1])
plt.show()
for m in range(102):
    psd_result = solos[0].spectrogram_psd[0, m, :, 20]
    plt.plot(psd_result)
plt.title("Density")
plt.xlabel("time")
plt.ylabel("density")
plt.show()