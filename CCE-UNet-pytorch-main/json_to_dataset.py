import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

if __name__ == '__main__':
    jpgs_path = "datasets/JPEGImages"
    pngs_path = "datasets/SegmentationClass"
    classes = ["background", "forest", "lake"]  # Modify "_background_" to "background"
    # classes = ["cat", "dog"]

    count = [f"{i}.json" for i in range(601, 701)]  # Assuming your json files are named as 601.json to 700.json
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                # Modify the following line to use the correct image file path
                imagePath = os.path.join("D:\\unet-pytorch-main\\datasets\\before", f"{i + 601}.jpg")
                if os.path.exists(imagePath):
                    with open(imagePath, 'rb') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode('utf-8')
                else:
                    print(f"Skipping {count[i]} due to missing image file: {imagePath}")
                    continue

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'background': 0}  # Modify "_background_" to "background"
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            PIL.Image.fromarray(img).save(osp.join(jpgs_path, f"{i + 601}.jpg"))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all * (np.array(lbl) == index_json)

            utils.lblsave(osp.join(pngs_path, f"{i + 601}.png"), new)
            print(f'Saved {i + 601}.jpg and {i + 601}.png')
