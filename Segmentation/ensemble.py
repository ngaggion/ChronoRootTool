import argparse
import pathlib
import re
import os
import numpy as np
import skimage.io as io

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext='*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key=natural_key)

    return all_files


def ensemble(path, alpha):
    images = loadPath(path, ext="*.png")
    img = io.imread(images[0], as_gray=True)
    accum = np.zeros(img.shape, dtype=np.float32)

    seg_path = os.path.join(path, "Segmentation")
    folds = [fold.split("/")[-1] for fold in loadPath(seg_path, ext="*/") if not fold.endswith(".png")]

    print("Ensembling {} folds.".format(len(folds)))
    print("Postprocessing with alpha = {}.".format(alpha))

    for image in images:
        if len(folds) >= 1:
            ensemble = np.zeros(img.shape, dtype=np.float32)
            for fold in folds:
                seg = io.imread(os.path.join(seg_path, fold, os.path.basename(image)), as_gray=True)
                seg = seg.astype(np.float32) / max(np.max(seg), 1)
                ensemble += seg
            ensemble /= len(folds)
        else:
            ensemble = io.imread(os.path.join(seg_path, os.path.basename(image)), as_gray=True)
            ensemble = ensemble.astype(np.float32) / max(np.max(ensemble), 1)

        accum = float(alpha) * accum + ensemble

        segmentation = accum > 0.5
        io.imsave(os.path.join(seg_path, os.path.basename(image)), segmentation.astype(np.uint8) * 255, check_contrast=False)

        del ensemble
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files and revert them back to original names.")
    parser.add_argument("path", help="Path to the folder containing the segmentation files.")
    parser.add_argument("--alpha", default=0.85, help="Weight for the temporal postprocessing step.")
    args = parser.parse_args()

    ensemble(args.path, args.alpha)