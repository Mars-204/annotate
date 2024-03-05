from PIL import Image
from pathlib import Path
import os
import shutil
import cv2
import random
# from groundingdino.util.inference import Model,load_image, predict, annotate
import os
import supervision as sv
# from autodistill_grounding_dino import GroundingDINO
# from autodistill.detection import CaptionOntology
import numpy as np

root_folder = Path(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\nikolaus\data_collection_anootated_inversed\data_collection\four_people')
annotation = Path(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\nikolaus\data_collection_anootated_inversed\data_collection\four_people\combined_0.25\annotations')
# files = list(root_folder.glob('*.txt'))
# for file in files:
#     with open(file, 'r+') as f:
#         if file.name == 'four_people_sv2_6042_00000000_comb.txt':
#             labelled_data = f.readlines()





# TEXT_PROMPT = "all persons"
# BOX_TRESHOLD = 0.35
# TEXT_TRESHOLD = 0.25
# base_model_dino = GroundingDINO(ontology=CaptionOntology({"all person": "person"}), box_threshold=0.25)
def get_cordinates(file, img):
    h, w = img.shape
    with open(file,'r+') as f:
        labelled_data = f.readlines()
        for data in labelled_data:
            data = data.split()
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])

            # Calculate the coordinates of the bounding box
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.imshow('Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


images = list(root_folder.glob('*inten.pgm'))
bboxes = list(annotation.glob('*'))
for _, image in enumerate(images):
    for _, bbox in enumerate(bboxes):
        if image.name[:-9] == bbox.name[12:30]:
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            get_cordinates(bbox, img)


    # image_source, im = load_image(image)
    # detection = base_model_dino.predict(image)
    # # boxes, logits, phrases = predict(
    # #     model=base_model_dino, 
    # #     image=im, 
    # #     caption=TEXT_PROMPT, 
    # #     box_threshold=BOX_TRESHOLD, 
    # #     text_threshold=TEXT_TRESHOLD
    # # )
    # boxes = detection.xyxy
    # logits = detection.confidence
    # phrases = detection.class_id
    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # sv.plot_image(annotated_frame, (16, 16))