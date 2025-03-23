from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    date_of_birth = Column(DateTime)
    is_active = Column(Boolean, default=True)
    is_doctor = Column(Boolean, default=False)
    
    health_data = relationship("HealthData", back_populates="user")
    ai_analyses = relationship("AIAnalysis", back_populates="user")

class HealthData(Base):
    __tablename__ = "health_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    data_type = Column(String)  # sleep, weight, diet, exercise, etc.
    data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="health_data")

class AIAnalysis(Base):
    __tablename__ = "ai_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    analysis_type = Column(String)  # health_profile, life_expectancy, action_plan
    analysis_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="ai_analyses")

class DoctorReview(Base):
    __tablename__ = "doctor_reviews"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"))
    analysis_id = Column(Integer, ForeignKey("ai_analyses.id"))
    review_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow) 