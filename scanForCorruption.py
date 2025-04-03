#!/usr/bin/env python3

import argparse
import os
import sys
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_image_corrupted(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return None  # Return None to indicate "no corruption"
    except Exception as e:
        return filepath  # Return the corrupted file path

def main():
    parser = argparse.ArgumentParser(description='Scan images for corruption.')
    parser.add_argument('folder', type=str, help='Folder path to scan.')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of parallel workers (default: 4).')
    args = parser.parse_args()

    # Gather all image paths
    image_paths = []
    for root, _, files in os.walk(args.folder):
        for file_name in files:
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
                image_paths.append(os.path.join(root, file_name))

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {executor.submit(is_image_corrupted, path): path for path in image_paths}

        for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc='Scanning images'):
            result = future.result()
            if result:  # i.e., corruption found
                print(f"Corrupted image found: {result}")

    print("No corrupted images found.")
    sys.exit(0)

if __name__ == '__main__':
    main()