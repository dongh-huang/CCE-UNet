import cv2
import numpy as np
import os
import random
# 定义数据增强函数
def random_rotation(image, label, max_angle=90, rows=None, cols=None):
    angle = random.randint(-max_angle, max_angle)
    if rows is None or cols is None:
        rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image_rotated = cv2.warpAffine(image, M, (cols, rows))
    label_rotated = cv2.warpAffine(label, M, (cols, rows))
    return image_rotated, label_rotated

def random_flip(image, label, flip_code=1, rows=None, cols=None):
    if rows is None or cols is None:
        rows, cols = image.shape[:2]
    image_flipped = cv2.flip(image, flip_code)
    label_flipped = cv2.flip(label, flip_code)
    return image_flipped, label_flipped

def random_zoom(image, label, zoom_range=(0.8, 1.2), rows=None, cols=None):
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
    if rows is None or cols is None:
        rows, cols, _ = image.shape
    M = np.float32([[zoom_factor, 0, (1-zoom_factor)/2 * cols],
                    [0, zoom_factor, (1-zoom_factor)/2 * rows]])
    image_zoomed = cv2.warpAffine(image, M, (cols, rows))
    label_zoomed = cv2.warpAffine(label, M, (cols, rows))
    return image_zoomed, label_zoomed

def random_crop(image, label, crop_range=(0.8, 1.0), rows=None, cols=None):
    crop_factor = random.uniform(crop_range[0], crop_range[1])
    if rows is None or cols is None:
        rows, cols, _ = image.shape
    crop_size = (int(cols * crop_factor), int(rows * crop_factor))
    x = random.randint(0, cols - crop_size[0])
    y = random.randint(0, rows - crop_size[1])
    image_cropped = image[y:y+crop_size[1], x:x+crop_size[0]]
    label_cropped = label[y:y+crop_size[1], x:x+crop_size[0]]
    return image_cropped, label_cropped

def random_translation(image, label, translation_range=(10, 10), rows=None, cols=None):
    tx = random.randint(-translation_range[0], translation_range[0])
    ty = random.randint(-translation_range[1], translation_range[1])
    if rows is None or cols is None:
        rows, cols, _ = image.shape
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    image_translated = cv2.warpAffine(image, M, (cols, rows))
    label_translated = cv2.warpAffine(label, M, (cols, rows))
    return image_translated, label_translated


# 定义批量数据增强函数
def augment_data(image, label, num_augmented_images_per_image=5):
    augmented_images = []
    augmented_labels = []
    augmentations = [
        random_rotation,
        random_flip,
        random_zoom,
        random_crop,
        random_translation
    ]
    for augmentation in augmentations:
        # 应用每种增强方法
        image_augmented, label_augmented = augmentation(image, label)

        # 保存增强后的图像和标签
        augmented_images.append(image_augmented)
        augmented_labels.append(label_augmented)
    return augmented_images, augmented_labels


# 定义批量数据增强函数
def augment_data(image, label, num_augmented_images_per_image=5):
    augmented_images = []
    augmented_labels = []
    augmentations = [
        random_rotation,
        random_flip,
        random_zoom,
        random_crop,
        random_translation
    ]

    for augmentation in augmentations:
        # 应用每种增强方法
        image_augmented, label_augmented = augmentation(image, label)

        # 保存增强后的图像和标签
        augmented_images.append(image_augmented)
        augmented_labels.append(label_augmented)

    return augmented_images, augmented_labels


# 读取图像和标签
# image_folder = r'C:\path\to\your\images'  # 替换为您的图像文件夹路径
# label_folder = r'C:\path\to\your\labels'  # 替换为您的标签文件夹路径
image_folder = r'C:\Users\16505\Desktop\Segmentation\JPEGImages'
label_folder = r'C:\Users\16505\Desktop\Segmentation\SegmentationClass'
# output_image_folder = r'C:\path\to\augmented\images'  # 替换为保存增强图像的路径
# output_label_folder = r'C:\path\to\augmented\labels'  # 替换为保存增强标签的路径
output_image_folder = r'C:\Users\16505\Desktop\Segmentation\jpg'
output_label_folder = r'C:\Users\16505\Desktop\Segmentation\png'

images = [cv2.imread(os.path.join(image_folder, filename)) for filename in sorted(os.listdir(image_folder))]
labels = [cv2.imread(os.path.join(label_folder, filename), 0) for filename in sorted(os.listdir(label_folder))]

# 确保图像和标签数量一致
assert len(images) == len(labels), "The number of images and labels must be equal."

# 执行数据增强并保存
for i, (image, label) in enumerate(zip(images, labels)):
    augmented_images, augmented_labels = augment_data(image, label)
    for j, (image_augmented, label_augmented) in enumerate(zip(augmented_images, augmented_labels)):
        cv2.imwrite(os.path.join(output_image_folder, f'image_{i}_{j}.jpg'), image_augmented)
        cv2.imwrite(os.path.join(output_label_folder, f'label_{i}_{j}.png'), label_augmented)
