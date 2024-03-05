from autodistill_grounded_sam import GroundedSAM
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from pathlib import Path
import os
import shutil
# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations


def annotator(root_folder, save_dir, sam=False, dino=True):
  if sam:
    base_model_sam = GroundedSAM(ontology=CaptionOntology({"all person": "person"}))
    base_model_sam.label(
        # input_folder=r"C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\manthan-test",
        input_folder="./images",
        output_folder="./dataset_sam",
        extension=".pgm"
      )
    
  folder_name = root_folder /  str(root_folder.name + "_intensity")  
  os.makedirs(folder_name, exist_ok=True)
  intenisty_images = list(root_folder.glob("*inten.pgm"))

  for im in intenisty_images:
    shutil.copy(im, folder_name)

  if dino:
    base_model_dino = GroundingDINO(ontology=CaptionOntology({"all person": "person"}), box_threshold=0.35)
    # label all images in a folder called `context_images`

    base_model_dino.label(input_folder=str(folder_name),
        output_folder=str(save_dir),
        extension=".pgm")
    
  print('Success')

# target_model = YOLOv8("yolov8n.pt")
# target_model.train("./dataset/data.yaml", epochs=200)

# run inference on the new model
# pred = target_model.predict("./dataset/valid/your-image.jpg", confidence=0.5)
# print(pred)

# optional: upload your model to Roboflow for deployment
# from roboflow import Roboflow

# rf = Roboflow(api_key="API_KEY")
# project = rf.workspace().project("PROJECT_ID")
# project.version(DATASET_VERSION).deploy(model_type="yolov8", model_path=f"./runs/detect/train/")

# base_model_dino = GroundingDINO(ontology=CaptionOntology({"all person": "person"}), box_threshold=0.2)
#     # label all images in a folder called `context_images`

# base_model_dino.label(input_folder='./images',
#     output_folder='./annotation',
#     extension=".pgm")
# base_model_dino.predict(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\nikolaus\output--sv2-lh004-objdet-06-person-agv\sv2_6043_00000426_inten.pgm')