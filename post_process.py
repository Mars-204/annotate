from pathlib import Path
import shutil
import os

root_folder = Path(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\data_collection\one_people_12122023\combined')
combined = list(root_folder.glob('*inten.txt'))

for file in combined:
    new_name = root_folder / str(file.stem[:-5] + "comb.txt")
    file.rename(new_name)
    