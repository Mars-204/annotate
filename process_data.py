import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from annotator import annotator

class PreProcess():
    """
    Class to combine image, generate diff image
    """
    def __init__(self, ref_img, image_folder) -> None:
        self.image_folder = image_folder
        self.ref_img = ref_img

    def modify_data(self, data, image_width, image_height, image_ID):
        data = data.split()
        if data:
            data = [float(x) for x in data]  # data["instance_id" "class_id", "xcorner_top", "ycorner_top", "width", "height"]

            data[2] = (data[2] + (data[4]/2))/image_width  # Xcenter = xcorner - width/2
            data[3] = (data[3] + (data[5]/2))/image_height  # Ycenter = ycorner + height/2
            data[4] = data[4]/image_width
            data[5] = data[5]/image_height
            del data[0]
            data[0] = int(data[0]) # converting float to int for object class
            data = [str(x) for x in data]
            data = " ".join(data)
        else:
            print(f'No labels for image ID {image_ID}')
        
        return data
    
    # def modify_diff(self):
    #     combined = self.image_folder / 'combined_diff'
    #     combined_images = combined.glob('*.npy')
    #     names = []
    #     for i in combined_images:
    #         names.append(i.name)
    #     images = self.image_folder.glob('*.npy')

    #     save_dir = self.image_folder / 'combined_diff_modified'
    #     os.makedirs(save_dir, exist_ok=True)
    #     for image in images:
    #         if image.name.split('_')[-3] == '6041':
    #             ref_image = self.ref_img[0]
    #         elif image.name.split('_')[-3] == '6042':
    #             ref_image = self.ref_img[1]
    #         elif image.name.split('_')[-3] == '6043':
    #             ref_image = self.ref_img[2]
            
    #         im = np.load(image)
    #         difference_img = self.diff_image(ref_image, im)
    #         difference_img_name = save_dir / image.name
    #         np.save(difference_img_name, difference_img)

            


    def process_data(self):
        # data_folder = Path(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\safe_visionary_T_mini')
        folder = self.image_folder
        # folder_list = list(data_folder.glob('*'))
        print("Creating npy images")
        save_dir = folder / 'diff'
        convert_npy = True
        if convert_npy:
            depth_images = list(folder.glob('*_depth.pgm'))
            intensity_images = list(folder.glob('*_inten.pgm'))
            labels = list(folder.glob('*.txt'))
            for j,_ in enumerate(depth_images):

                if depth_images[j].name.split('_')[1] == '7060':
                    ref_image = self.ref_img[0]
                elif depth_images[j].name.split('_')[1] == '7061':
                    ref_image = self.ref_img[1]
                elif depth_images[j].name.split('_')[1] == '6043':
                    ref_image = self.ref_img[2]
                
                if depth_images[j].name.split('_')[-2] == intensity_images[j].name.split('_')[-2]:
                    depth_img = cv2.imread(str(depth_images[j]), cv2.IMREAD_ANYDEPTH)
                    intensity_img = cv2.imread(str(intensity_images[j]), cv2.IMREAD_ANYDEPTH)
                    image_height, image_width = depth_img.shape

                    combined_img = np.dstack((intensity_img, depth_img))
                    os.makedirs(save_dir, exist_ok=True)
                    combined_img_name = str(save_dir / str(folder.stem + '_'+ depth_images[j].stem[:-5])) + 'comb.npy'
                    
                    combined_img = self.diff_image(ref_image, combined_img)
                    np.save(combined_img_name, combined_img)
        print("Generated npy images. Starting annotation")
        
        #Annotate data:
        annotator(self.image_folder,save_dir)
        annotation_folder = save_dir / "annotations"
        combined = list(annotation_folder.glob('*inten.txt'))

        for file in combined:
            new_name = annotation_folder / str(folder.stem + '_' + file.stem[:-5] +'comb.txt')
            file.rename(new_name)
                # shutil.copy(str(labels[j]), processed_folder)
                # label_file = processed_folder / labels[j].name
                
                # image_ID = labels[j].name.split('_')[-2]
                # new_file_name = str(processed_folder / labels[j].stem[:-4]) + 'comb.txt'
                # # if image_ID in ['922', '294', '412', '694', '971', '417', '78', '922']:
                # with open(label_file, 'r') as input_file:
                #     output_txt = new_file_name
                #     with open(output_txt, 'w') as output_file:
                #         for line in input_file:
                #             data = self.modify_data(line, image_width, image_height, image_ID)
                #             output_file.write(data + '\n')
                # os.remove(label_file)

                            
    def combine_image(self):
        images_folder = Path(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\image-files-test')
        depth_images = list(images_folder.glob('depth*'))
        intensity_images = list(images_folder.glob('inten*'))
        processed_folder = images_folder / 'combined_diff'
        os.makedirs(processed_folder, exist_ok=True)

        ## obtain reference numpy image
        ref_depth_image_path = r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\two_person_06122023\sv2_6042_00000000_depth.pgm'
        ref_inten_image_path = r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\two_person_06122023\sv2_6042_00000000_inten.pgm'
        ref_depth_img = cv2.imread(ref_depth_image_path, cv2.IMREAD_ANYDEPTH)
        ref_depth_img = ref_depth_img.astype(np.uint8)
        ref_intensity_img = cv2.imread(ref_inten_image_path, cv2.IMREAD_ANYDEPTH)
        ref_combined_img = np.dstack((ref_intensity_img, ref_depth_img))
        for _, depth in enumerate(depth_images):
            for _, intensity in enumerate(intensity_images):
                if depth.stem.split('_')[-1] == intensity.stem.split('_')[-1]:
                    depth_img = cv2.imread(str(depth), cv2.IMREAD_ANYDEPTH)
                    intensity_img = cv2.imread(str(intensity), cv2.IMREAD_ANYDEPTH)
                    depth_img = depth_img.astype(np.uint8)
                    combined_img = np.dstack((intensity_img, depth_img))
                    combined_img = self.diff_image(ref_combined_img, combined_img)
                    combined_img_name = str(processed_folder / intensity.stem.split('_')[-1]) + '_comb.npy'
                    
                    np.save(combined_img_name, combined_img)
                    break

    def diff_image(self, ref_img, img):
        """
        Difference of two numpy images
        """
        diff_img = np.zeros((424, 512, 2), dtype=np.uint16)
        diff = cv2.absdiff(ref_img[:,:,1], img[:,:,1])

        kernel = np.ones((6,6), np.uint16)
        diff_erode = cv2.erode(diff, kernel, iterations = 1)

        diff_img[:,:,1] = np.where(diff_erode>150, img[:,:,1], 0)
        diff_img[:,:,0] = np.where(diff_erode>150, img[:,:,0], 0)

        # diff_img[:,:,0] = img[:,:,0]
        # image_normalized = cv2.normalize(diff_img[:,:,1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # color_image = cv2.applyColorMap(image_normalized, cv2.COLORMAP_HOT)
        # cv2.imshow('diff', diff_img[:,:,1])
        # # time.sleep(0.5)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()
        # fig.add_subplot(2, 2, 4) # first subplot
        # plt.imshow(diff_img[:,:,1]) # display first image
        # plt.title("final difference image") # set title
        # plt.axis("off") # turn off axis
        return diff_img
    
def gen_ref_img():
    # ref_img_6041_depth = cv2.imread(str(list(ref_folder.glob('sv2_6041_*depth.pgm'))[0]), cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(str(list(ref_folder.glob('sv2_6042_*depth.pgm'))[11]), cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(str(list(ref_folder.glob('sv2_6043_*depth.pgm'))[11]), cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(str(list(ref_folder.glob('sv2_6041_*depth.pgm'))[11]), cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(str(list(ref_folder.glob('sv2_6042_*depth.pgm'))[11]), cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(str(list(ref_folder.glob('sv2_6043_*depth.pgm'))[11]), cv2.IMREAD_ANYDEPTH)

    # # one_person
    # ref_img_6041_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\one_people_12122023\sv2_6041_00000199_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\one_people_12122023\sv2_6042_00000235_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\one_people_12122023\sv2_6043_00000213_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\one_people_12122023\sv2_6041_00000199_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\one_people_12122023\sv2_6042_00000235_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\one_people_12122023\sv2_6043_00000213_inten.pgm', cv2.IMREAD_ANYDEPTH)

    # # output_sv2....01 
    # ref_img_6041_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-01\sv2_6041_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-01\sv2_6042_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-01\sv2_6041_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-01\sv2_6042_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_inten.pgm', cv2.IMREAD_ANYDEPTH)    

    # # output_sv2....02
    # ref_img_6041_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-02-dark\sv2_6041_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-02-dark\sv2_6042_00000216_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-02-dark\sv2_6041_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-02-dark\sv2_6042_00000216_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_inten.pgm', cv2.IMREAD_ANYDEPTH)    

    # # output_sv2....03
    # ref_img_6041_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-03-obstructions-reference\sv2_6041_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-03-obstructions-reference\sv2_6042_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-03-obstructions-reference\sv2_6041_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-03-obstructions-reference\sv2_6042_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_inten.pgm', cv2.IMREAD_ANYDEPTH)    

    # # output_sv2....04
    # ref_img_6041_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-04-obstructions\sv2_6041_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-04-obstructions\sv2_6042_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-04-obstructions\sv2_6041_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-04-obstructions\sv2_6042_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_inten.pgm', cv2.IMREAD_ANYDEPTH)    

    # # output_sv2....05
    # ref_img_6041_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6041_00000067_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6042_00000178_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6041_00000067_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6042_00000178_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_inten.pgm', cv2.IMREAD_ANYDEPTH)    

    # output_sv2....06
    # ref_img_6041_depth = cv2.imread(r'D:\work\Nikolaus\four_people\four_people_12122023\sv2_6041_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'D:\work\Nikolaus\four_people\four_people_12122023\sv2_6042_00000157_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'D:\work\Nikolaus\four_people\four_people_12122023\sv2_6043_00000349_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'D:\work\Nikolaus\four_people\four_people_12122023\sv2_6041_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'D:\work\Nikolaus\four_people\four_people_12122023\sv2_6042_00000157_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'D:\work\Nikolaus\four_people\four_people_12122023\sv2_6043_00000349_inten.pgm', cv2.IMREAD_ANYDEPTH)    

    # # four_people
    # ref_img_6041_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\four_people\sv2_6041_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\four_people_reference_6042.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\four_people_reference_6043.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\four_people\sv2_6041_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-06-person-agv\sv2_6042_00000157_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-05-twopersonsmeet\sv2_6043_00000349_inten.pgm', cv2.IMREAD_ANYDEPTH)    

    # # two_people
    # ref_img_6041_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\two_person_06122023\sv2_6041_00000084_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\two_people_reference_6042.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_depth = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\two_person_06122023\sv2_6043_00000124_depth.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6041_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\two_person_06122023\sv2_6041_00000084_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6042_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\output--sv2-lh004-objdet-06-person-agv\sv2_6042_00000157_inten.pgm', cv2.IMREAD_ANYDEPTH)
    # ref_img_6043_inten = cv2.imread(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\two_person_06122023\sv2_6043_00000124_inten.pgm', cv2.IMREAD_ANYDEPTH)    
    
    ref_img_6041_depth = cv2.imread(r'D:\work\Nikolaus\23jan2024\01-reference\sv2_7060_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    ref_img_6042_depth = cv2.imread(r'D:\work\Nikolaus\23jan2024\01-reference\sv2_7061_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    ref_img_6043_depth = cv2.imread(r'D:\work\Nikolaus\23jan2024\01-reference\sv2_7060_00000000_depth.pgm', cv2.IMREAD_ANYDEPTH)
    ref_img_6041_inten = cv2.imread(r'D:\work\Nikolaus\23jan2024\01-reference\sv2_7060_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    ref_img_6042_inten = cv2.imread(r'D:\work\Nikolaus\23jan2024\01-reference\sv2_7061_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)
    ref_img_6043_inten = cv2.imread(r'D:\work\Nikolaus\23jan2024\01-reference\sv2_7060_00000000_inten.pgm', cv2.IMREAD_ANYDEPTH)  

    ref_img_6041 = np.dstack((ref_img_6041_inten, ref_img_6041_depth))
    ref_img_6042 = np.dstack((ref_img_6042_inten, ref_img_6042_depth))
    ref_img_6043 = np.dstack((ref_img_6043_inten, ref_img_6043_depth))
    print('reference images generated')
    ref = ref_img_6041, ref_img_6042, ref_img_6043
    return ref

def invert_image(image_folder_path):
    _images = list(image_folder_path.glob('*.pgm'))
    for image in _images:
        if not image.is_dir():
            im = Image.open(image)
            inverted_image = im.transpose(method=Image.ROTATE_180)
            save_inverted_dir = image_folder_path / "inverted_images"
            os.makedirs(save_inverted_dir, exist_ok=True)
            save_image = save_inverted_dir / str(str(image.stem) + '.pgm')
            inverted_image.save(save_image)
    print('Inverted all the images')
    return save_inverted_dir

if __name__ == '__main__':
    root = Path(r'D:\work\Nikolaus\23jan2024')
    ref_folder = Path(r'D:\work\Nikolaus\23jan2024')
    print("Generating reference images")
    ref_img = gen_ref_img()
    # image_folder = Path(r'C:\work\masterarbiet\3d-object-detection-and-tracking-using-dl\data\data_collection\manthan-test')
    # invert_image(image_folder)

    
    
    # save_folder = Path(r'P:\OT\Absolventen&Praktika\Diplomarbeiten_und_Praktika\_Patel\Nikolaus\output--sv2-lh004-objdet-06-person-agv\combined')
    # annotator(image_folder, save_folder)
    # process = PreProcess(ref_img, image_folder)
    # process.process_data()
    data_folders = list(root.glob('*'))
    for folder in data_folders:
        if folder.is_dir():
            print(folder)
            # inverted_image_folder = invert_image(folder)
            process = PreProcess(ref_img, folder)
            process.process_data()

    # process = PreProcess(ref_img, ref_folder)
    # process.process_data()
    # process.modify_diff()