def convert_to_gsn_hydrocel_names(raw):
    """
    Rename EEG channels from 'EEG X' format to 'EX' (GSN-HydroCel).
    VREF is renamed to 'Cz'.
    
    Parameters:
    - raw: mne.io.Raw - The raw EEG data
    
    Returns:
    - mne.io.Raw - The raw data with renamed channels
    """
    mapping = {}
    for ch in raw.ch_names:
        if ch.upper() == 'EEG VREF':
            mapping[ch] = 'Cz'
        elif ch.startswith('EEG '):
            try:
                num = int(ch.split(' ')[1])
                mapping[ch] = f"E{num}"
            except:
                print(f"Skipping unrecognized channel: {ch}")
    raw.rename_channels(mapping)
    return raw