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