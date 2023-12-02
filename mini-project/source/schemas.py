from pydantic import BaseModel
from typing import List


class SegmentationPrediction(BaseModel):
    segmentation_prediction: List[List[int]]
