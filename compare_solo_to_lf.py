from qtmWrapper.qtm import QTM
import smoothnessMeasurement

# TODO: Fill in the paths
solo_file_path = ""
lf_file_path = ""

#### QTM configuration ####
load_from_dump = True
skeleton = False
markers_permutation_vector = True
smoothing_factor = 1500
fix_jumps = True

def get_dump_path(file_path):
    return file_path.split(".")[0] + "_interpolated_mirrored_qtm.npz"

def get_mean_scores(qtm_obj, marker):
    mean_sparc_score = smoothnessMeasurement.get_mean_sparc_for_all_segments(qtm_obj, object=0, marker=marker)
    # TODO: Change 'object' to 0 or 1 (depending on the object you want to measure)
    mean_jerk_score = smoothnessMeasurement.get_mean_jerk_for_all_segments(qtm_obj, object=0, marker=marker)
    return mean_sparc_score, mean_jerk_score

def main():
    solo_qtm_obj = QTM(solo_file_path, save_path=get_dump_path(solo_file_path), load=load_from_dump, 
                       num_of_subjects=1, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
    lf_qtm_obj = QTM(lf_file_path, save_path=get_dump_path(lf_file_path), load=load_from_dump,
                        num_of_subjects=2, markers_permutation_vector=markers_permutation_vector, skeleton=skeleton, fix_jumps=fix_jumps, interpolate=True, smoothing_factor=smoothing_factor)
    solo_scores = get_mean_scores(solo_qtm_obj, marker=0)
    lf_scores = get_mean_scores(lf_qtm_obj, marker=0)
    print(f"solo scores: {solo_scores}")
    print(f"lf scores: {lf_scores}")

if __name__ == '__main__':
    main()