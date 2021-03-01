import cv2
import os
import glob
import numpy as np
from shutil import copyfile

ho3d_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Data', 'datasets', 'HO3D_V2'))
ho3d_path_iCWT = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Data', 'datasets', 'HO3D_V2_iCWT_format'))
if not os.path.exists(ho3d_path_iCWT):
    os.mkdir(ho3d_path_iCWT)

ho3d_path_train = os.path.abspath(os.path.join(ho3d_path, 'train'))
ho3d_path_train_iCWT = os.path.abspath(os.path.join(ho3d_path_iCWT, 'train'))
if not os.path.exists(ho3d_path_train_iCWT):
    os.mkdir(ho3d_path_train_iCWT)

target_images_dir = os.path.abspath(os.path.join(ho3d_path_train_iCWT, 'Images'))
if not os.path.exists(target_images_dir):
    os.mkdir(target_images_dir)
target_annotations_dir = os.path.abspath(os.path.join(ho3d_path_train_iCWT, 'Annotations'))
if not os.path.exists(target_annotations_dir):
    os.mkdir(target_annotations_dir)
target_masks_dir = os.path.abspath(os.path.join(ho3d_path_train_iCWT, 'Masks'))
if not os.path.exists(target_masks_dir):
    os.mkdir(target_masks_dir)
target_imagesets_dir = os.path.abspath(os.path.join(ho3d_path_train_iCWT, 'ImageSets'))
if not os.path.exists(target_imagesets_dir):
    os.mkdir(target_imagesets_dir)


dirs_list = sorted(glob.glob(os.path.abspath(os.path.join(ho3d_path_train, '*'))))

dir_names_to_class ={
    'ABF10': '021_bleach_cleanser',
    'ABF11': '021_bleach_cleanser',
    'ABF12': '021_bleach_cleanser',
    'ABF13': '021_bleach_cleanser',
    'ABF14': '021_bleach_cleanser',
    'BB10': '011_banana',
    'BB11': '011_banana',
    'BB12': '011_banana',
    'BB13': '011_banana',
    'BB14': '011_banana',
    'GPMF10': '010_potted_meat_can',
    'GPMF11': '010_potted_meat_can',
    'GPMF12': '010_potted_meat_can',
    'GPMF13': '010_potted_meat_can',
    'GPMF14': '010_potted_meat_can',
    'GSF10': '037_scissors',
    'GSF11': '037_scissors',
    'GSF12': '037_scissors',
    'GSF13': '037_scissors',
    'GSF14': '037_scissors',
    'MC1': '003_cracker_box',
    'MC2': '003_cracker_box',
    'MC4': '003_cracker_box',
    'MC5': '003_cracker_box',
    'MC6': '003_cracker_box',
    'MDF10': '035_power_drill',
    'MDF11': '035_power_drill',
    'MDF12': '035_power_drill',
    'MDF13': '035_power_drill',
    'MDF14': '035_power_drill',
    'ND2': '035_power_drill',
    'SB10': '021_bleach_cleanser',
    'SB12': '021_bleach_cleanser',
    'SB14': '021_bleach_cleanser',
    'ShSu10': '004_sugar_box',
    'ShSu12': '004_sugar_box',
    'ShSu13': '004_sugar_box',
    'ShSu14': '004_sugar_box',
    'SiBF10': '011_banana',
    'SiBF11': '011_banana',
    'SiBF12': '011_banana',
    'SiBF13': '011_banana',
    'SiBF14': '011_banana',
    'SiS1': '004_sugar_box',
    'SM2': '006_mustard_bottle',
    'SM3': '006_mustard_bottle',
    'SM4': '006_mustard_bottle',
    'SM5': '006_mustard_bottle',
    'SMu1': '025_mug',
    'SMu40': '025_mug',
    'SMu41': '025_mug',
    'SMu42': '025_mug',
    'SS1': '004_sugar_box',
    'SS2': '004_sugar_box',
    'SS3': '004_sugar_box'
}

def compute_annotations_xml(path_to_annotations_dir, img_name, objects_data):

    # Retrieving annotations from file
    target_file = img_name + '.xml'
    target_path = os.path.join(path_to_annotations_dir, target_file)
    target = open(target_path, 'w')

    target.write('<annotation>')

    target.write('\n\t<folder>')
    target.write('Images')
    target.write('</folder>')

    target.write('\n\t<filename>')
    target.write(img_name)
    target.write('</filename>')

    target.write('\n\t<source>')

    target.write('\n\t\t<database>HO3D_V2</database>')

    target.write('\n\t</source>')

    target.write('\n\t<size>')

    target.write('\n\t\t<width>640</width>')

    target.write('\n\t\t<height>480</height>')

    target.write('\n\t\t<depth>3</depth>')

    target.write('\n\t</size>')

    target.write('\n\t<tstamp>')
    target.write('0')
    target.write('</tstamp>')

    target.write('\n\t<segmented>1</segmented>')

    for obj in objects_data:

        if not obj['label'] == 'dummy':

            target.write('\n\t<object>')

            target.write('\n\t\t<category>')
            target.write(obj['category'])
            target.write('</category>')

            target.write('\n\t\t<name>')
            target.write(obj['label'])
            target.write('</name>')

            target.write('\n\t\t<truncated>0</truncated>')

            target.write('\n\t\t<difficult>0</difficult>')

            target.write('\n\t\t<bndbox>')

            target.write('\n\t\t\t<xmin>')
            target.write(obj['xmin'])
            target.write('</xmin>')

            target.write('\n\t\t\t<ymin>')
            target.write(obj['ymin'])
            target.write('</ymin>')

            target.write('\n\t\t\t<xmax>')
            target.write(obj['xmax'])
            target.write('</xmax>')

            target.write('\n\t\t\t<ymax>')
            target.write(obj['ymax'])
            target.write('</ymax>')

            target.write('\n\t\t</bndbox>')

            target.write('\n\t</object>')

    target.write('\n</annotation>')
    target.close()

for dir in dirs_list:
    print(dir)
    for k, v in dir_names_to_class.items():
        if k in dir:
            obj_class = v
            obj_dir = k
            break
    segm_dir = os.path.abspath(os.path.join(dir, 'seg'))
    binary_segm_dir = os.path.abspath(os.path.join(target_masks_dir, obj_dir))
    if not os.path.exists(binary_segm_dir):
        os.mkdir(binary_segm_dir)

    annotations_dir = os.path.abspath(os.path.join(target_annotations_dir, obj_dir))
    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)

    images_dir = os.path.abspath(os.path.join(target_images_dir, obj_dir))
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    segm_files = sorted(glob.glob(os.path.abspath(os.path.join(segm_dir, '*'))))
    for segm_file in segm_files:
        img_name = segm_file.replace(segm_dir, '').replace('.png', '').replace('.jpg', '').replace('/', '')

        mask = cv2.resize(cv2.imread(segm_file), (640, 480))
        obj_indices = np.where((mask >= [100, 0, 0]).all(axis=2))

        # Compute segmentation binary masks
        binary_mask = np.zeros([mask.shape[0], mask.shape[1], 1], dtype=np.uint8)
        for i in range(len(obj_indices[0])):
            binary_mask[obj_indices[0][i], obj_indices[1][i]] = 255

        cv2.imwrite(os.path.abspath(os.path.join(binary_segm_dir, img_name + '.png')), binary_mask)

        # Compute annotations
        objects = []

        if len(obj_indices[0]) > 0:  #Check that the object is visible
            object = {}
            object['xmin'] = str(min(obj_indices[1]))
            object['ymin'] = str(min(obj_indices[0]))
            object['xmax'] = str(max(obj_indices[1])+1)
            object['ymax'] = str(max(obj_indices[0])+1)
            object['label'] = obj_class
            object['category'] = obj_class
            objects.append(object)
        compute_annotations_xml(annotations_dir, img_name, objects)

        src_img_dir = os.path.abspath(os.path.join(dir, 'rgb', '%s.png'))
        # Copy images
        copyfile(src_img_dir%img_name, os.path.abspath(os.path.join(target_images_dir, obj_dir, img_name + '.png')))
