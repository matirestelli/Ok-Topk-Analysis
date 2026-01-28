#!/usr/bin/env python3
"""
Script to regenerate AN4 manifest files with correct paths
"""
import os
import glob

# Base path to your AN4 dataset
base_path = os.path.join(os.path.dirname(__file__), 'audio_data', 'an4_dataset')
output_dir = os.path.join(os.path.dirname(__file__), 'audio_data')

def generate_manifest(split='train'):
    """Generate manifest file for train or test split"""
    
    wav_dir = os.path.join(base_path, split, 'an4', 'wav')
    txt_dir = os.path.join(base_path, split, 'an4', 'txt')
    
    if not os.path.exists(wav_dir):
        print(f"Warning: {wav_dir} does not exist")
        return
    
    # Get all wav files
    wav_files = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
    
    manifest_lines = []
    for wav_file in wav_files:
        # Get the base name without extension
        basename = os.path.basename(wav_file)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Create corresponding txt file path
        txt_file = os.path.join(txt_dir, name_without_ext + '.txt')
        
        # Check if txt file exists
        if os.path.exists(txt_file):
            # Use absolute paths
            manifest_lines.append(f"{os.path.abspath(wav_file)},{os.path.abspath(txt_file)}")
        else:
            print(f"Warning: No transcript found for {wav_file}")
    
    # Write manifest
    if split == 'train':
        manifest_path = os.path.join(output_dir, 'an4_train_manifest.csv')
    else:
        manifest_path = os.path.join(output_dir, 'an4_val_manifest.csv')
    
    with open(manifest_path, 'w') as f:
        f.write('\n'.join(manifest_lines) + '\n')
    
    print(f"Generated {manifest_path} with {len(manifest_lines)} entries")

if __name__ == '__main__':
    print(f"Base dataset path: {base_path}")
    generate_manifest('train')
    generate_manifest('test')  # Note: test split, but written to an4_val_manifest.csv
    print("Manifest files regenerated successfully!")
