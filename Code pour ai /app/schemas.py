from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    date_of_birth: datetime
    is_doctor: bool = False

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class HealthDataBase(BaseModel):
    data_type: str
    data: Dict[str, Any]

class HealthDataCreate(HealthDataBase):
    user_id: int

class HealthData(HealthDataBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class AIAnalysisBase(BaseModel):
    analysis_type: str
    analysis_data: Dict[str, Any]

class AIAnalysisCreate(AIAnalysisBase):
    user_id: int

class AIAnalysis(AIAnalysisBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class DoctorReviewBase(BaseModel):
    review_data: Dict[str, Any]

class DoctorReviewCreate(DoctorReviewBase):
    doctor_id: int
    analysis_id: int

class DoctorReview(DoctorReviewBase):
    id: int
    doctor_id: int
    analysis_id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None 