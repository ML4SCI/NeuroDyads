import numpy as np
import mne

def mutually_align_cuts(raw1, raw2):
    """
    Align two EEG recordings by mutually identifying and removing cut segments.
    
    Parameters:
    - raw1: mne.io.Raw - First EEG recording
    - raw2: mne.io.Raw - Second EEG recording
    
    Returns:
    - tuple: (raw1_clean, raw2_clean) - Cleaned and aligned raw objects
    """
    sfreq = raw1.info['sfreq']
    data1 = raw1.get_data().copy()
    data2 = raw2.get_data().copy()

    # 1. Pad the shorter file with zeros
    max_len = max(data1.shape[1], data2.shape[1])
    if data1.shape[1] < max_len:
        pad = np.zeros((data1.shape[0], max_len - data1.shape[1]))
        data1 = np.concatenate([data1, pad], axis=1)
    if data2.shape[1] < max_len:
        pad = np.zeros((data2.shape[0], max_len - data2.shape[1]))
        data2 = np.concatenate([data2, pad], axis=1)

    # 2. Insert zeros based on annotations
    def insert_zeros(data, annotations):
        for onset, duration in zip(annotations.onset, annotations.duration):
            start = int(onset * sfreq)
            stop = int((onset + duration) * sfreq)
            data[:, start:stop] = 0
        return data

    data1 = insert_zeros(data1, raw1.annotations)
    data2 = insert_zeros(data2, raw2.annotations)

    # 3. Identify zero regions
    zero_mask = np.all(data1 == 0, axis=0) | np.all(data2 == 0, axis=0)

    # 4. Find cut segments
    zero_diff = np.diff(np.concatenate(([0], zero_mask.astype(int), [0])))
    start_idxs = np.where(zero_diff == 1)[0]
    end_idxs = np.where(zero_diff == -1)[0]

    # 5. Create annotations
    new_annotations = []
    for start, end in zip(start_idxs, end_idxs):
        onset = start / sfreq
        duration = (end - start) / sfreq
        new_annotations.append((onset, duration))

    # 6. Define keep segments
    keep_segments = []
    current = 0
    for start, end in zip(start_idxs, end_idxs):
        if start > current:
            keep_segments.append((current, start))
        current = end
    if current < max_len:
        keep_segments.append((current, max_len))

    # 7. Rebuild both files
    def extract(data):
        return np.concatenate([data[:, s:e] for s, e in keep_segments], axis=1)

    new_data1 = extract(data1)
    new_data2 = extract(data2)

    # 8. Create new Raw objects
    raw1_clean = mne.io.RawArray(new_data1, raw1.info.copy())
    raw2_clean = mne.io.RawArray(new_data2, raw2.info.copy())

    # 9. Add shared annotations
    annots = mne.Annotations(
        onset=[a[0] for a in new_annotations],
        duration=[a[1] for a in new_annotations],
        description=['cut'] * len(new_annotations)
    )
    raw1_clean.set_annotations(annots)
    raw2_clean.set_annotations(annots)

    return raw1_clean, raw2_clean