import mne
from mne.preprocessing import ICA
from mne.channels import read_custom_montage
from pyprep.find_noisy_channels import NoisyChannels
import os
import json
from datetime import datetime
import zipfile
from .rename_channels import convert_to_gsn_hydrocel_names

def preprocess_eeg(input_data, name=None, montage_path='GSN-HydroCel-65_1.0.sfp', output_dir='preprocessed_eeg/'):
    """
    Preprocess EEG data from a .set file or a Raw object and save as .edf after ICA.
    
    Parameters:
    - input_data: str or mne.io.Raw - path to EEGLAB .set file or already loaded Raw object
    - name: str - optional name used for output files when a Raw object is passed
    - montage_path: str - path to the montage .sfp file
    - output_dir: str - directory where the output EDF and log will be saved
    
    Returns:
    - str: Path to the saved preprocessed file
    """
    os.makedirs(output_dir, exist_ok=True)

    processing_log = {
        'timestamp': datetime.now().isoformat(),
        'processing_steps': []
    }

    # Load data
    if isinstance(input_data, str):
        raw = mne.io.read_raw_eeglab(input_data, preload=True)
        processing_log['original_file'] = input_data
        base = os.path.splitext(os.path.basename(input_data))[0]
        processing_log['processing_steps'].append({
            'step': 'load_data',
            'description': f'Loaded EEG data from {input_data}',
            'n_channels': len(raw.ch_names),
            'duration_seconds': raw.times[-1]
        })
    elif isinstance(input_data, mne.io.BaseRaw):
        raw = input_data
        processing_log['original_file'] = 'Raw object provided directly'
        base = name if name else "raw_input"
        processing_log['processing_steps'].append({
            'step': 'use_raw_object',
            'description': 'Used already loaded Raw object',
            'n_channels': len(raw.ch_names),
            'duration_seconds': raw.times[-1]
        })
    else:
        raise ValueError("input_data must be a filepath string or an mne.io.Raw object")

    # Rename channels and set montage
    raw = convert_to_gsn_hydrocel_names(raw)
    montage = read_custom_montage(montage_path)
    raw.set_montage(montage)
    processing_log['processing_steps'].append({
        'step': 'set_montage',
        'description': f'Applied montage from {montage_path}',
        'montage_channels': montage.ch_names
    })

    # High-pass filter
    raw.filter(l_freq=1.0, h_freq=None)
    processing_log['processing_steps'].append({
        'step': 'highpass_filter',
        'description': 'Applied 1.0 Hz high-pass filter',
        'filter_settings': {'l_freq': 1.0, 'h_freq': None}
    })

    # Automatic bad channel detection using pyprep
    nc = NoisyChannels(raw, random_state=1337)
    nc.find_all_bads()

    bad_channels = {
        'bad_by_nan': nc.bad_by_nan,
        'bad_by_flat': nc.bad_by_flat,
        'bad_by_deviation': nc.bad_by_deviation,
        'bad_by_hf_noise': nc.bad_by_hf_noise,
        'bad_by_correlation': nc.bad_by_correlation,
        'bad_by_ransac': nc.bad_by_ransac
    }

    all_bads = list(set(
        nc.bad_by_nan +
        nc.bad_by_flat +
        nc.bad_by_deviation +
        nc.bad_by_hf_noise +
        nc.bad_by_correlation +
        nc.bad_by_ransac
    ))

    raw.info['bads'] = all_bads
    processing_log['processing_steps'].append({
        'step': 'bad_channel_detection',
        'description': 'Identified bad channels using pyprep',
        'bad_channels': bad_channels,
        'all_bad_channels': all_bads,
        'n_bad_channels': len(all_bads)
    })

    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    processing_log['processing_steps'].append({
        'step': 'interpolate_bads',
        'description': 'Interpolated bad channels',
        'interpolated_channels': all_bads
    })

    # Ask user how many ICA components to use
    while True:
        try:
            n_components = int(input(f"Enter the number of ICA components to compute (recommended ≤ {len(raw.ch_names)}): "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Run ICA
    ica = ICA(n_components=n_components, random_state=97, max_iter='auto')
    ica.fit(raw)
    processing_log['processing_steps'].append({
        'step': 'ica_fit',
        'description': 'Fitted ICA components',
        'ica_settings': {
            'n_components': n_components,
            'random_state': 97,
            'max_iter': 'auto'
        }
    })

    # Plot ICA components and save all figures as PNGs, then zip them
    print("Plotting ICA components and saving images...")
    ica_figs = ica.plot_components(show=True)
    
    png_paths = []

    for i, fig in enumerate(ica_figs):
        png_path = os.path.join(output_dir, f"{base}_ica_components_page{i+1}.png")
        fig.savefig(png_path, dpi=300)
        png_paths.append(png_path)

    zip_path = os.path.join(output_dir, f"{base}_ica_components.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for png_file in png_paths:
            zipf.write(png_file, arcname=os.path.basename(png_file))

    processing_log['processing_steps'].append({
        'step': 'ica_plot',
        'description': 'Saved ICA components figure for visual inspection',
        'n_components_plotted': n_components
    })

    # Ask user for components to exclude
    to_exclude = input("Enter ICA component numbers to exclude (comma-separated): ")
    to_exclude = [int(i.strip()) for i in to_exclude.split(',') if i.strip().isdigit()]
    ica.exclude = to_exclude
    processing_log['processing_steps'].append({
        'step': 'ica_component_selection',
        'description': 'User-selected ICA components to exclude',
        'excluded_components': to_exclude,
        'n_components_excluded': len(to_exclude)
    })

    # Apply ICA
    raw = ica.apply(raw.copy())
    processing_log['processing_steps'].append({
        'step': 'ica_apply',
        'description': 'Applied ICA cleaning',
        'n_components_removed': len(to_exclude)
    })

    # Save output
    output_path = os.path.join(output_dir, f"{base}_preprocessed.edf")
    mne.export.export_raw(output_path, raw, fmt='edf', overwrite=True)

    # Save processing log
    log_path = os.path.join(output_dir, f"{base}_processing_log.json")
    with open(log_path, 'w') as f:
        json.dump(processing_log, f, indent=4)

    print(f"Preprocessed file saved to: {output_path}")
    print(f"Processing log saved to: {log_path}")
    return output_path