from fastapi import APIRouter, UploadFile, Depends, HTTPException
from source.services import ImgTesseract
import PIL

router = APIRouter()


@router.post("/coordinates")
async def get_bboxes_coordinates(file: UploadFile, img_tesseract: ImgTesseract = Depends(ImgTesseract)):
    """
    Extract the bounding boxes coordinates of the text lines in the image file.

    :param file: An image file uploaded using `multipart/form-data` encoding.
    :param img_tesseract: An instance of the `ImgTesseract` class.
    :return: A dictionary containing the coordinates of the bounding boxes of each text line in the image file.
    """
    try:
        result = await img_tesseract.bboxes_coordinates(file)
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return result


@router.post("/image-bounding-boxes")
async def visualize_bboxes(file: UploadFile, img_tesseract: ImgTesseract = Depends(ImgTesseract)):
    """
    Visualize the bounding boxes of text lines in the image file.
    :param file: An image file uploaded using `multipart/form-data` encoding.
    :param img_tesseract: An instance of the `ImgTesseract` class.
    :return: The image with bounding boxes drawn around the text lines.
    """
    try:
        result = await img_tesseract.visualize_bboxes(file)
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return result


@router.post("/extract-text")
async def extract_text(file: UploadFile, img_tesseract: ImgTesseract = Depends(ImgTesseract)):
    """
    Extract the text from the image file using Tesseract-OCR Engine.
    :param file: An image file uploaded using `multipart/form-data` encoding.
    :param img_tesseract: An instance of the `ImgTesseract` class.
    :return: The extracted text from the image.
    """
    try:
        result = img_tesseract.image_to_text(file)
    except PIL.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return result
