import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from io import BytesIO
from PIL import Image


class Model:
    def __init__(self, modeldir):
        self.model = tf.saved_model.load(modeldir)

    def predict(self, image, output_path):
        image, height, width = self.__preprocess_img(image)
        return Prediction(self.model, image, output_path, height, width)

    def __preprocess_img(self, image):
        image = tf.image.decode_image(image, channels=3)

        width, height, _ = image.shape

        image = tf.image.resize(image, [640,640])
        image = image / 255.
        image = tf.expand_dims(image, axis=0)

        return image, height, width

class Prediction:
    def __init__(self, model : Model, image : np.ndarray, output_path, height, width, iou_threshold=0.7, prob_threshold=0.25, mask_threshold=0.25):
        self.__model = model
        self.image = image
        self.width = width
        self.height = height
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.mask_threshold = mask_threshold
        self.output_path = output_path
        self.result = self.__predict()

    def __predict(self):
        output = self.__model(self.image)
        return self.__output_preprocessing(output)

    def __output_preprocessing(self, model_output):
        # Apply non max suppression
        valid_bboxes = self.__nms(model_output, self.prob_threshold, self.iou_threshold)

        prototypes = model_output[1].numpy().squeeze().transpose(2, 0, 1)
        coeffs = valid_bboxes[:, 5:]

        # Assemble protomasks
        assembly = self.__sigmoid(-coeffs @ prototypes.reshape(32, 160*160))
        assembly = assembly.reshape(-1, 160, 160)
        
        # Apply upscale
        assembly = tf.image.resize(np.expand_dims(assembly, axis=-1), (self.width, self.height), method=tf.image.ResizeMethod.LANCZOS3).numpy().squeeze()

        masks = self.__crop_masks(assembly, valid_bboxes)
        masks = self.__thresholding(masks, self.mask_threshold)

        cpas = np.count_nonzero(masks, axis=(1,2))
        global_mask = np.amax(masks, axis=0)

        output_image = global_mask.copy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)

        for x1, y1, x2, y2 in valid_bboxes[:, :4].astype(int):
            output_image = cv2.rectangle(output_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        cv2.imwrite(self.output_path, output_image)

        return (valid_bboxes.tolist(), cpas.tolist(), masks.tolist(), global_mask.tolist())

    def __sigmoid(self, array : np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(array)) 

    def __convert_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        xc = bboxes[:, 0].copy()
        yc = bboxes[:, 1].copy()
        half_width = bboxes[:, 2] / 2
        half_height = bboxes[:, 3] / 2
        bboxes[:, 0] = xc - half_width
        bboxes[:, 1] = yc - half_height
        bboxes[:, 2] = xc + half_width
        bboxes[:, 3] = yc + half_height
        return bboxes

    def __box_area(self, bboxes : np.ndarray) -> float:
        return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    def __intersection_area(self, bboxes : np.ndarray) -> np.ndarray:
        top_left = np.maximum(bboxes[:, None, :2], bboxes[:, :2])
        bottom_right = np.minimum(bboxes[:, None, 2:4], bboxes[:, 2:4])

        return np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), axis=2)

    def __nms(self, predictions, prob_treshold=0.25, iou_threshold=0.7):
        # Get bounding boxes and probabilities
        bboxes = tf.reshape(tf.transpose(predictions[0]), shape=(8400, 37)).numpy()

        # Filter and sort
        filter_idx = bboxes[:, 4] > prob_treshold
        bboxes = bboxes[filter_idx]
        bboxes = bboxes[bboxes[:, 4] > prob_treshold]

        sort_idx = np.flip(bboxes[:, 4].argsort())

        bboxes = bboxes[sort_idx]
        # bboxes = bboxes[:, :4]
        bboxes = self.__convert_bboxes(bboxes)
        
        # Get box area and intersection area
        box_areas = self.__box_area(bboxes)
        inter_areas = self.__intersection_area(bboxes)

        # Calculate iou
        ious = inter_areas / (box_areas[:, None] + box_areas - inter_areas)

        # Discard self iou
        ious = ious - np.eye(ious.shape[0])
        
        keep = np.ones(ious.shape[0], dtype=bool)

        for index, iou in enumerate(ious):
            if not keep[index]:
                continue

            condition = (iou > iou_threshold)
            keep = keep & ~condition

        return bboxes[keep]
    
    def __crop_masks(self, assembled_mask : np.ndarray, bboxes : np.ndarray) -> np.ndarray :
        height = assembled_mask.shape[-2]
        width = assembled_mask.shape[-1]

        new_mask = assembled_mask.reshape(-1, height, width).copy()
        keep_mask = np.zeros(new_mask.shape, dtype=bool)

        # Create filters
        for idx, (x1, y1, x2, y2) in enumerate(bboxes[:, :4].astype(int)):
            keep_mask[idx, y1:y2+1, x1:x2+1] = True

        # Apply filters
        new_mask = new_mask * keep_mask

        return new_mask

    def __thresholding(self, masks : np.ndarray, threshold = 0.25) -> np.ndarray :
        return masks > threshold




    


    