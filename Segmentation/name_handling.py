import argparse
import pathlib
import re
import os

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext='*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key=natural_key)

    return all_files

def rename_files(path):
    files = loadPath(path, ext="*.png")
    name_mapping = {}
    counter = 0

    for file in files:
        file_path = pathlib.Path(file)
        new_name = file_path.parent / f"image_{counter:03d}_0000.png"
        os.rename(file, new_name)
        name_mapping[new_name.name] = file_path.name
        counter += 1

    return name_mapping

def revert_file_names(path, name_mapping):
    for new_name, original_name in name_mapping.items():
        new_path = pathlib.Path(path) / new_name
        original_path = pathlib.Path(path) / original_name
        os.rename(new_path, original_path)

def revert_seg_file_names(path, name_mapping):
    for new_name, original_name in name_mapping.items():
        new_name = new_name.replace("_0000.png", ".png")
        new_path = pathlib.Path(path) / new_name
        original_path = pathlib.Path(path) / original_name
        os.rename(new_path, original_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files and revert them back to original names.")
    parser.add_argument("path", help="Path to the folder containing files to be renamed.")
    parser.add_argument("--revert", action="store_true", help="Revert file names to original names.")
    parser.add_argument("--revert_seg", action="store_true", help="Revert segmentation file names to original names.")
    parser.add_argument("--segpath", default = "None", help="Path to the segmentation folder containing files to be renamed.")
    args = parser.parse_args()

    if args.revert:
        # Load name mapping from a file
        with open(os.path.join(args.path, "name_mapping.txt"), "r") as f:
            name_mapping = {line.split(',')[0]: line.split(',')[1].strip() for line in f}
        revert_file_names(args.path, name_mapping)
    elif args.revert_seg:
        # Load name mapping from a file
        with open(os.path.join(args.path, "name_mapping.txt"), "r") as f:
            name_mapping = {line.split(',')[0]: line.split(',')[1].strip() for line in f}
        revert_seg_file_names(args.segpath, name_mapping)
    else:
        name_mapping = rename_files(args.path)
        # Save name mapping to a file
        with open(os.path.join(args.path, "name_mapping.txt"), "w") as f:
            for new_name, original_name in name_mapping.items():
                f.write(f"{new_name},{original_name}\n")