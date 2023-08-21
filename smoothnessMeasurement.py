from matplotlib import pyplot as plt
import numpy as np
import math
from mathModels.smoothness_metrics import sparc, dimensionless_jerk, log_dimensionless_jerk

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

    def print_smoothness_scores(self):
        print(f"SPARC: {self.get_sparc()[0]}")
        print(f"Dimensionless jerk: {self.get_dimensionless_jerk()}")
        print(f"Log dimensionless jerk: {self.get_log_dimensionless_jerk()}")
    
    def get_scores_by_marker(self, marker):
        data = self.get_speed_profille_by_marker(marker)
        return self.get_sparc(data)[0], self.get_dimensionless_jerk(data), self.get_log_dimensionless_jerk(data)

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
        plt.plot(self.speed_profile, label="Speed profile")
        plt.show()
    
    def plot_grid_of_speed_profile_of_markers(self):
        fig, axs = plt.subplots(3, 3)
        for i, ax in enumerate(axs.flat):
            ax.plot(self.qtm_obj.velocities[self.object, i, :, self.start_frame:self.end_frame].T)
            ax.set_title(f"Marker {i}", fontsize=10)
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
    
    def plot_sparc_as_func_of_amp_th_param(self):
        _, ax = plt.subplots()
        amp_th_range = np.arange(0, 1, 0.01)
        scores = []
        for i in amp_th_range:
            scores.append(self.get_sparc(amp_th=i)[0])
        ax.scatter(amp_th_range, scores)
        ax.set_title(f"Sparc score as a function of amplitude threshold (segement {self.segment_start}-{self.segment_end})", **self.title_params)
        ax.set_xlabel("Amplitude Threshold", fontsize=self.fontsize-4)
        ax.set_ylabel("Score", fontsize=self.fontsize-4)
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
def get_scores_for_all_segments(qtm_obj, object, marker, segment_size=30, segment_margin=0, sample_freq=500, padlevel=4, freq_cutoff=5, amp_th=0.05):
    scores = {"sparc": [], "jerk": [], "log_jerk": []}
    total_size = len(qtm_obj.data[object, marker, 0, :])
    # Segments of segment_size seconds
    number_of_segments = math.ceil((total_size / qtm_obj.frame_rate) / segment_size) - (segment_margin * 2)
    # print(f"total size (num of frames): {total_size}, number of segments: {number_of_segments})")
    smoothness_comparator = SmoothnessMeasurement(qtm_obj, object, marker, sample_freq=sample_freq, padlevel=padlevel, freq_cutoff=freq_cutoff, amp_th=amp_th)
    # print(f"ITERATING FROM {segment_margin} TO {number_of_segments * segment_size} (jumps of {segment_size})")
    for i, segment_start in enumerate(np.arange(segment_margin, number_of_segments * segment_size, segment_size)):
        # print(f"Seg number {i}: {segment_start} - {segment_start + segment_size}")
        smoothness_comparator.set_segment_start(segment_start)
        smoothness_comparator.set_segment_end(segment_start + segment_size)
        if segment_start + segment_size > number_of_segments * segment_size:
            # print(f"breaking at {segment_start} because {segment_start + segment_size} > {number_of_segments * segment_size}")
            break
        try:
            if smoothness_comparator.speed_profile.shape[0] == 0:
                print("continuing")
                continue
            sparc_score_of_seg = smoothness_comparator.get_sparc()[0]
            jerk_score_of_seg = smoothness_comparator.get_dimensionless_jerk()
            log_jerk_score_of_seg = smoothness_comparator.get_log_dimensionless_jerk()
            scores["sparc"].append(sparc_score_of_seg)
            scores["jerk"].append(jerk_score_of_seg)
            scores["log_jerk"].append(log_jerk_score_of_seg)
            # print(f"Scores for segment {segment_start} - {segment_start + segment_size}: sparc {sparc_score_of_seg}, jerk {jerk_score_of_seg}")
        except ZeroDivisionError:
            print(f"got zero division error for segment {segment_start}-{segment_start+segment_size}")
            break
    # print(f"returning scores: {scores}")
    return scores
    # return np.mean(scores["sparc"]), np.mean(scores["jerk"])