# YOLOv8 Yield Sage Model

YOLOv8 Instance Segmentation Model trained with our dataset exported into TensorFlow saved_model format.

## Example Usage

```py
from model import Model

model = Model('path to saved model')
result = model.predict('path to image', 'output path')

```