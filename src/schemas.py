from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class ClinicalEvent(BaseModel):
    entity_name: str = Field(description="Normalized name of drug, diagnosis, lab, or allergy (e.g., 'Warfarin', 'Penicillin')")
    category: Literal["Medication", "Diagnosis", "LabResult", "Allergy", "Procedure"]
    status: Literal["Active", "Started", "Dose_Changed", "Held", "Discontinued", "Suspected", "Ruled_Out", "Abnormal"]
    value_or_dose: Optional[str] = Field(default=None, description="Dosage (e.g., '5mg daily') or lab value (e.g., 'INR 4.8')")
    event_date: str = Field(description="ISO date YYYY-MM-DD when the clinical event occurred")
    doc_id: str = Field(description="Exact ID of the source document (e.g., 'DOC_01_ADMISSION')")
    exact_quote: str = Field(description="Verbatim sentence or phrase from the text proving this fact")

class DocumentExtraction(BaseModel):
    doc_id: str
    doc_type: Literal["Admission_Note", "Progress_Note", "Lab_Report", "Discharge_Summary", "Prescription"]
    doc_date: str
    events: List[ClinicalEvent]

class ConflictAlert(BaseModel):
    severity: Literal["CRITICAL", "WARNING"]
    conflict_type: Literal["Outdated_Medication", "Allergy_Mismatch", "Contradictory_Diagnosis", "Missing_Followup"]
    description: str
    evidence_a: str = Field(description="Quote + Doc ID of first conflicting record")
    evidence_b: str = Field(description="Quote + Doc ID of second conflicting record")

class AuditResult(BaseModel):
    conflicts: List[ConflictAlert]
    missing_information: List[str]
    trust_score: float = Field(description="0.0 = Completely untrustworthy, 1.0 = Fully verified")
    must_escalate: bool = Field(description="True if AI must STOP and escalate to a human doctor")
    escalation_reason: Optional[str] = None