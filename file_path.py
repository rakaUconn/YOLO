# import os
#
# testing_images_path = "C:/underwater_object_detection/Data/MangoYOLO/VOCDevkit/VOC2007/JPEGImages/"
# filenames = [
#     "gear8n_(16)_03_03.jpg", "gear8n_(16)_03_04.jpg"]
#
# for filename in filenames:
#     full_path = os.path.join(testing_images_path, filename)
#     if not os.path.exists(full_path):
#         print(f"File not found: {full_path}")
#     else:
#         print(f"File exists: {full_path}")

import tensorflow as tf

def load_image_with_bboxes(image_path, bboxes):
    if not tf.io.gfile.exists(image_path):
        tf.print(f"File does not exist: {image_path}")
        return None, None
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [640, 640])
        image = tf.cast(image, tf.float32) / 255.0
        return image, bboxes
    except Exception as e:
        tf.print(f"Error loading image {image_path}: {e}")
        return None, None

def parse_function(example):
    image_path = example['image_path']
    bboxes = example['bboxes']
    image, bboxes = load_image_with_bboxes(image_path, bboxes)
    if image is None:
        return None
    return {'images': image, 'bounding_boxes': bboxes}

def create_dataset(file_paths, bboxes):
    dataset = tf.data.Dataset.from_tensor_slices({'image_path': file_paths, 'bboxes': bboxes})
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x: x is not None)  # Remove None entries
    return dataset

# Example file paths and bounding boxes
file_paths = [
    "C:/underwater_object_detection/Data/Mango_data/Image/gear1n_(1)_02_03.jpg",
    "C:/underwater_object_detection/Data/Mango_data/Image/gear1n_(6)_03_03.png",  # Example of a missing file
    # Add more file paths here...
]
bboxes = [
    # Add corresponding bounding boxes here...
]

# Create dataset
train_ds = create_dataset(file_paths, bboxes)
