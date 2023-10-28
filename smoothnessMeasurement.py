from matplotlib import pyplot as plt
import numpy as np
import math
from mathModels.smoothness_metrics import sparc, dimensionless_jerk, log_dimensionless_jerk, velocity_peaks_per_meter, speed_metric

class SmoothnessMeasurement:
    
    def __init__(self, qtm_obj, object, marker, segment_start=None, segment_end=None, 
                 frame_margin=1000, sample_freq=500, padlevel=4, freq_cutoff=5, amp_th=0.05):
        self.qtm_obj = qtm_obj
        self.object = object
        self.marker = marker
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.frame_margin = frame_margin
        self.sample_freq = sample_freq
        self.padlevel = padlevel
        self.freq_cutoff = freq_cutoff
        self.amp_th = amp_th
        self.title_params = dict(fontsize=8, loc="left")
        self.labelsize = 8
        self.fontsize = 10

    @property
    def start_frame(self):
        return int(self.segment_start * self.qtm_obj.frame_rate)

    @property
    def end_frame(self):
        return int(self.segment_end * self.qtm_obj.frame_rate)

    @property
    def speed_profile(self):
        return np.linalg.norm(self.qtm_obj.velocities[self.object, self.marker, :, self.start_frame:self.end_frame], axis=0)
    
    def set_segment_start(self, new_start):
        self.segment_start = new_start

    def set_segment_end(self, new_end):
        self.segment_end = new_end

    def get_speed_profille_by_marker(self, marker):
        return np.linalg.norm(self.qtm_obj.velocities[self.object, marker, :, self.start_frame:self.end_frame], axis=0)
    
    def get_sparc(self, data=None, amp_th=None):
        data = self.speed_profile if data is None else data
        amp_th = amp_th or self.amp_th
        return sparc(data, fs=self.sample_freq, fc=self.freq_cutoff, padlevel=self.padlevel, amp_th=amp_th)
    
    def get_dimensionless_jerk(self, data=None):
        data = self.speed_profile if data is None else data
        return dimensionless_jerk(data, fs=self.sample_freq)

    def get_log_dimensionless_jerk(self, data=None):
        data = self.speed_profile if data is None else data
        return log_dimensionless_jerk(self.speed_profile, fs=self.sample_freq)
    
    def get_velocity_peaks_per_meter(self, data=None):
        data = self.speed_profile if data is None else data
        return velocity_peaks_per_meter(data, fs=self.sample_freq)

    def get_speed_metric(self, data=None):
        data = self.speed_profile if data is None else data
        return speed_metric(data)

    def print_smoothness_scores(self):
        print(f"SPARC: {self.get_sparc()[0]}")
        print(f"Dimensionless jerk: {self.get_dimensionless_jerk()}")
        print(f"Log dimensionless jerk: {self.get_log_dimensionless_jerk()}")
        print(f"Velocity peaks per meter: {self.get_velocity_peaks_per_meter()}")
        print(f"Speed metric: {self.get_speed_metric()}")
    
    def plot_positions(self):
        fig, axes = plt.subplots(3, 1)
        title_params = dict(fontsize=8, loc="left")
        labelsize = 8
        fontsize = 10
        coordinate_labels = ["X", "Y", "Z"]
        for (i, ax) in enumerate(axes.flat):
            ax.plot(self.qtm_obj.time[self.start_frame:self.end_frame], self.qtm_obj.data[self.object, self.marker, i, self.start_frame:self.end_frame])
            ax.set_title(f"Coordinate {coordinate_labels[i]}", **title_params)
            ax.set_ylabel("Position", fontsize=fontsize-2)
            ax.tick_params(axis='both', which='major', labelsize=labelsize)
        axes[2].set_xlabel("Time [Sec]", fontsize=fontsize-2)
        plt.show()

    def plot_speed_profile(self):
        title_params = dict(fontsize=10, loc="left")
        plt.plot(self.qtm_obj.time[self.start_frame:self.end_frame], self.speed_profile)
        plt.title(f"Speed Profile (Segment {self.segment_start}-{self.segment_end})", **title_params)
        plt.xlabel("Time [Sec]", fontsize=self.fontsize-2)
        plt.ylabel("Speed", fontsize=self.fontsize-2)
        plt.show()
    
    def plot_frequency_and_magnitude(self):
        _, fmf, fmf_sel = self.get_sparc()
        f, mf = fmf
        f_sel, mf_sel = fmf_sel
        _, ax = plt.subplots()
        ax.plot(f, mf, label="magnitude spectrum", color="blue")
        ax.plot(f_sel, mf_sel, label="selected spectrum", color="red")
        ax.set_title(f"Speed profile magnitude spectrum with amp_th={self.amp_th}", **self.title_params)
        ax.set_xlabel("frequency [hz]", fontsize=self.fontsize-2)
        ax.set_ylabel("magnitude", fontsize=self.fontsize-2)
        ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
        ax.legend()
        plt.show()
    
def plot_all_coords_by_marker(marker, qtm_obj, object, segment_margin, file_name=""):
    frame_margin = int(segment_margin * qtm_obj.frame_rate)
    fig, axes = plt.subplots(3, 1)
    title_params = dict(fontsize=8, loc="left")
    labelsize = 8
    fontsize = 10
    coordinate_labels = ["X", "Y", "Z"]
    for (i, ax) in enumerate(axes.flat):
        ax.plot(qtm_obj.time[frame_margin:-frame_margin], qtm_obj.data[object, marker, i, frame_margin:-frame_margin])
        ax.set_title(f"Marker {marker}, Object {object}: Coordinate {coordinate_labels[i]} ({file_name})", **title_params)
        ax.set_ylabel("Position", fontsize=fontsize-2)
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
    axes[2].set_xlabel("Time [Sec]", fontsize=fontsize-2)
    plt.show()

"""
Returns the mean scores of sparc and jerk calculated on the data without segment_margin at beggining and end in segments of segment_size.
"""
def get_scores_for_segments_by_segment_size(qtm_obj, object, marker, segment_size=30, segment_margin=0, sample_freq=500, padlevel=4, freq_cutoff=5, amp_th=0.05):
    scores = {"sparc": [], "jerk": [], "log_jerk": [], "velocity_peaks": [], "mean_speed": []}
    total_size = len(qtm_obj.data[object, marker, 0, :])
    # Segments of segment_size seconds
    number_of_segments = math.ceil((total_size / qtm_obj.frame_rate) / segment_size) - (segment_margin * 2)
    smoothness_comparator = SmoothnessMeasurement(qtm_obj, object, marker, sample_freq=sample_freq, padlevel=padlevel, freq_cutoff=freq_cutoff, amp_th=amp_th)
    for i, segment_start in enumerate(np.arange(segment_margin, number_of_segments * segment_size, segment_size)):
        smoothness_comparator.set_segment_start(segment_start)
        smoothness_comparator.set_segment_end(segment_start + segment_size)
        if segment_start + segment_size > number_of_segments * segment_size:
            break
        try:
            if smoothness_comparator.speed_profile.shape[0] == 0:
                continue
            sparc_score_of_seg = smoothness_comparator.get_sparc()[0]
            jerk_score_of_seg = smoothness_comparator.get_dimensionless_jerk()
            log_jerk_score_of_seg = smoothness_comparator.get_log_dimensionless_jerk()
            velocity_peaks_of_seg = smoothness_comparator.get_velocity_peaks_per_meter()
            mean_speed_of_seg = smoothness_comparator.get_speed_metric()
            scores["sparc"].append(sparc_score_of_seg)
            scores["jerk"].append(jerk_score_of_seg)
            scores["log_jerk"].append(log_jerk_score_of_seg)
            scores["velocity_peaks"].append(velocity_peaks_of_seg)
            scores["mean_speed"].append(mean_speed_of_seg)
        except ZeroDivisionError:
            print(f"got zero division error for segment {segment_start}-{segment_start+segment_size}")
            break
    return scores

def run_analytics_on_syntetic_data():
    import numpy as np
    sampling_rate = 1000
    duration = 5
    frequency_low = 0.5 # Hz
    frequency_high = 2.5 # Hz

    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    low_frequency_signal = np.sin(2 * np.pi * frequency_low * t)
    high_frequency_signal = np.sin(2 * np.pi * frequency_high * t)
    synthetic_signal = low_frequency_signal + high_frequency_signal

    # Plot the synthetic signal and print its smoothness metrics scores
    plt.figure(figsize=(10, 6))
    plt.plot(t, synthetic_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Synthetic Data with Low and High Frequencies')
    plt.grid(True)
    plt.show()

    sal, _, _ = sparc(synthetic_signal, sampling_rate)
    dl = dimensionless_jerk(synthetic_signal, sampling_rate)
    ldl = log_dimensionless_jerk(synthetic_signal, sampling_rate)
    vp = velocity_peaks_per_meter(synthetic_signal, sampling_rate)
    speed = speed_metric(synthetic_signal)
    print("Low and High Frequency Signal")
    print('Spectral Arc Length: %.5f' % sal)
    print('Dimensionless Jerk: %.5f' % dl)
    print('Log Dimensionless Jerk: %.5f' % ldl)
    print('Number of Velocity Peaks per Meter: %.5f' % vp)
    print('Speed Metric: %.5f' % speed)
    print()
    
    # Plot and print for low frequency signal
    plt.figure(figsize=(10, 6))
    plt.plot(t, low_frequency_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Low Frequency Signal')
    plt.grid(True)
    plt.show()

    sal, _, _ = sparc(low_frequency_signal, sampling_rate)
    dl = dimensionless_jerk(low_frequency_signal, sampling_rate)
    ldl = log_dimensionless_jerk(low_frequency_signal, sampling_rate)
    vp = velocity_peaks_per_meter(low_frequency_signal, sampling_rate)
    speed = speed_metric(low_frequency_signal)
    print("Low Frequency Signal")
    print('Spectral Arc Length: %.5f' % sal)
    print('Dimensionless Jerk: %.5f' % dl)
    print('Log Dimensionless Jerk: %.5f' % ldl)
    print('Number of Velocity Peaks per Meter: %.5f' % vp)
    print('Speed Metric: %.5f' % speed)
    print()

    # Plot and print for high frequency signal
    plt.figure(figsize=(10, 6))
    plt.plot(t, high_frequency_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('High Frequency Signal')
    plt.grid(True)
    plt.show()

    sal, _, _ = sparc(high_frequency_signal, sampling_rate)
    dl = dimensionless_jerk(high_frequency_signal, sampling_rate)
    ldl = log_dimensionless_jerk(high_frequency_signal, sampling_rate)
    vp = velocity_peaks_per_meter(high_frequency_signal, sampling_rate)
    speed = speed_metric(high_frequency_signal)
    print("High Frequency Signal")
    print('Spectral Arc Length: %.5f' % sal)
    print('Dimensionless Jerk: %.5f' % dl)
    print('Log Dimensionless Jerk: %.5f' % ldl)
    print('Number of Velocity Peaks per Meter: %.5f' % vp)
    print('Speed Metric: %.5f' % speed)
    print()
