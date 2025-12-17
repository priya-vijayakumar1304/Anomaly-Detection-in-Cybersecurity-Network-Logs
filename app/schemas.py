from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    total_records: int
    anomalies_detected: int
    normal_records: int
