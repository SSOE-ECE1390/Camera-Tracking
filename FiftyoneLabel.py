import fiftyone as fo
from pathlib import Path
import glob as glob

RAW_DIR = r"C:\Users\colto\OneDrive - University of Pittsburgh\Documents\GitHub\Camera-Tracking\CNN Work\redcar_data\images_all"

# create a FiftyOne dataset from the folder
dataset = fo.Dataset("redcar_dataset")
dataset.add_samples([fo.Sample(filepath=str(p)) for p in Path(RAW_DIR).glob("*.jpg")])

# launch GUI
session = fo.launch_app(dataset)
session.wait()

dataset.export(
    export_dir="redcar_labeled",           # folder where YOLO labels/images will be saved
    dataset_type=fo.types.YOLOv5Dataset    # YOLO format
)