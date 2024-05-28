import xml.etree.ElementTree as ET
import os
from tqdm.auto import tqdm
def edit_xml_file(xml_file,folder):

    dir_name,file_namep = os.path.split(xml_file)
    name_ext,_ = os.path.splitext(file_namep)
    new_file_name = name_ext + '.jpg'
    dir_f, file_nf = os.path.split(dir_name)
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    first_child = root.find("path")
    if first_child is not None:
        first_child.text = dir_f + '/' + folder + '/' + new_file_name
    second_child = root.find("folder")
    if second_child is not None:
        second_child.text = folder
    t_child = root.find("filename")
    if t_child is not None:
        t_child.text = new_file_name
    # Write the changes back to the file
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)




path_annot = "C:/underwater_object_detection/Data/person_yolo/Train/Train/JPEGImages/"

xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
    ]
)


for xml_file in tqdm(xml_files):
    edit_xml_file(xml_file,'JPEGImages')