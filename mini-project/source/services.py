import io
import numpy as np
import cv2
from pytesseract import Output
import pytesseract
from PIL import Image
from fastapi import Response, UploadFile
from configuration.config import pytesseract_config
from source.schemas import SegmentationPrediction


class ImgTesseract:
    @staticmethod
    def get_info(image):
        """
        Get the pytesseract dictionary from image and the mask of lines indexes
        :param image: the image to be processed
        :return: A tuple containing the pytesseract dictionary and a list of line indices.
        """
        pytess_dict = pytesseract.image_to_data(image, output_type=Output.DICT, lang=pytesseract_config.text_lang)
        lines = [i for i in range(len(pytess_dict['level'])) if
                 pytess_dict['level'][i] == 4]
        return pytess_dict, lines

    async def bboxes_coordinates(self, file: UploadFile):
        """
        Extract the bounding boxes coordinates of the text lines in the image file.
        :param file: An image file uploaded using `multipart/form-data` encoding.
        :return:  dictionary containing the coordinates of the bounding boxes of each text line
        detected in the image
        """
        image = Image.open(io.BytesIO(await file.read()))
        pytess_dict, lines = self.get_info(image)
        coordinates = [
            [pytess_dict['left'][line], pytess_dict['top'][line], pytess_dict['width'][line],
             pytess_dict['height'][line]] for line in
            lines]
        return SegmentationPrediction(segmentation_prediction=coordinates)

    async def visualize_bboxes(self, file: UploadFile):
        """
        Visualize the bounding boxes of text lines in the image file.
            :param file: An image file uploaded using `multipart/form-data` encoding.
            :return: displays the image with bounding boxes drawn on top.
            """
        image = Image.open(io.BytesIO(await file.read()))
        img = np.array(image)
        rgb_img = cv2.cvtColor(img,
                               cv2.COLOR_BGR2RGB)
        pytess_dict, lines = self.get_info(image)
        for line in lines:
            (x, y, w, h) = (
                pytess_dict['left'][line], pytess_dict['top'][line], pytess_dict['width'][line],
                pytess_dict['height'][line])
            cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _, png = cv2.imencode('.png', rgb_img)
        return Response(content=png.tobytes(), media_type='image/png')

    @staticmethod
    def image_to_text(file: UploadFile):
        """
        Extract the text from the image file using Tesseract-OCR Engine.
        :param file: An image file uploaded using `multipart/form-data` encoding.
        :return: extracted test from the original image.
        """
        image = Image.open(file.file)
        text = pytesseract.image_to_string(image, lang=pytesseract_config.text_lang)
        return {"text": text}
