import mne
from preprocessing.rename_channels import convert_to_gsn_hydrocel_names
from preprocessing.mutual_alignment import mutually_align_cuts
from preprocessing.individual_preprocessing import preprocess_eeg
import argparse

def main():
    parser = argparse.ArgumentParser(description='EEG Processing Pipeline')
    parser.add_argument('file1', help='First EEG file (.set)')
    parser.add_argument('file2', help='Second EEG file (.set)')
    parser.add_argument('--montage', default='GSN-HydroCel-65_1.0.sfp', 
                       help='Path to montage file')
    parser.add_argument('--output_dir', default='preprocessed_output',
                       help='Output directory for processed files')
    
    args = parser.parse_args()

    # Step 1: Load and rename channels
    print("Loading and renaming channels...")
    raw1 = mne.io.read_raw_eeglab(args.file1, preload=True)
    raw2 = mne.io.read_raw_eeglab(args.file2, preload=True)
    
    raw1 = convert_to_gsn_hydrocel_names(raw1)
    raw2 = convert_to_gsn_hydrocel_names(raw2)

    # Step 2: Mutual alignment
    print("Performing mutual alignment...")
    raw1_aligned, raw2_aligned = mutually_align_cuts(raw1, raw2)

    # Step 3: Individual preprocessing
    print("Preprocessing first file...")
    output1 = preprocess_eeg(
        raw1_aligned, 
        name="file1_preprocessed",
        montage_path=args.montage,
        output_dir=args.output_dir
    )

    print("Preprocessing second file...")
    output2 = preprocess_eeg(
        raw2_aligned,
        name="file2_preprocessed",
        montage_path=args.montage,
        output_dir=args.output_dir
    )

    print("\nProcessing complete!")
    print(f"Output files saved to: {output1} and {output2}")

if __name__ == "__main__":
    main()