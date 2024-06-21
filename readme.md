# YOLOv8 Yield Sage Model

YOLOv8 Instance Segmentation Model trained with our dataset exported into TensorFlow saved_model format.

## Example Usage
```py
from model import Model
import tensorflow as tf

model = Model('path to saved model')
image = tf.io.read_file('path to image')
result = model.predict(image, 'output path')
```

## Sample output
- Input image
  ![image](https://github.com/capstone-product-bangkit/yolov8-yieldsage/assets/114855785/745a82ee-35a8-4e52-a9a1-5ce28302724d)

- Output image
  ![image](https://github.com/capstone-product-bangkit/yolov8-yieldsage/assets/114855785/b4f7aa4b-4ee5-43e7-9562-9c3dbe1c4484)

We also have bounding box outputs, crown projection areas, individual masks, and global mask which is shown as the output image above
  

