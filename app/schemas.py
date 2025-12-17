from pydantic import BaseModel
from typing import List, Dict, Any

class PredictionResponse(BaseModel):
    total_records: int
    anomalies_detected: int
    normal_records: int
    data: List[Dict[str, Any]]

