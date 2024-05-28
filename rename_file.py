
#source_file = 'C:/underwater_object_detection/Data/MangoYOLO/VOCDevkit/VOC2007/JPEGImages/old_name.jpg'
import os
import shutil

#
# def rename_and_move_jpg_files(src_folder, dst_folder, new_name_base):
#     # Ensure the destination folder exists
#     if not os.path.exists(dst_folder):
#         os.makedirs(dst_folder)
#
#     # Get a list of all jpg files in the source folder
#     jpg_files = [f for f in os.listdir(src_folder) if f.lower().endswith('.jpg')]
#
#     for i, filename in enumerate(jpg_files):
#         # Construct new file name
#         new_name = f"{new_name_base}_{i + 1}.jpg"
#
#         # Full paths for source and destination
#         src_path = os.path.join(src_folder, filename)
#         dst_path = os.path.join(dst_folder, new_name)
#
#         # Move and rename the file
#         shutil.move(src_path, dst_path)
#         print(f"Moved: {src_path} to {dst_path}")
#
#
# # Example usage:
# src_folder = 'C:/underwater_object_detection/Data/MangoYOLO/VOCDevkit/VOC2007/JPEGImages'
# dst_folder = 'C:/underwater_object_detection/Data/Mango_data/Image'
# new_name_base = 'image'
#
# rename_and_move_jpg_files(src_folder, dst_folder, new_name_base)



def rename_and_move_xml_files(src_folder, dst_folder, new_name_base):
    # Ensure the destination folder exists
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Get a list of all jpg files in the source folder
    jpg_files = [f for f in os.listdir(src_folder) if f.lower().endswith('.xml')]

    for i, filename in enumerate(jpg_files):
        # Construct new file name
        new_name = f"{new_name_base}_{i + 1}.xml"

        # Full paths for source and destination
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, new_name)

        # Move and rename the file
        shutil.move(src_path, dst_path)
        print(f"Moved: {src_path} to {dst_path}")


# Example usage:
src_folder = 'C:/underwater_object_detection/Data/MangoYOLO/VOCDevkit/VOC2007/Annotations'
dst_folder = 'C:/underwater_object_detection/Data/Mango_data/Annotation'
new_name_base = 'image'

rename_and_move_xml_files(src_folder, dst_folder, new_name_base)