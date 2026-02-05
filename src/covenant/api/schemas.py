from pydantic import BaseModel


class ActionRequest(BaseModel):
    mission: str
    confidence: float
    evidence: float
    harm: float
    resource_ratio: float
    auditable: bool
    data_provenance: bool
