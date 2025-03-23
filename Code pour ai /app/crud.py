from sqlalchemy.orm import Session
from . import models, schemas
from .auth import get_password_hash

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        date_of_birth=user.date_of_birth,
        is_doctor=user.is_doctor
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_health_data(db: Session, health_data: schemas.HealthDataCreate):
    db_health_data = models.HealthData(**health_data.dict())
    db.add(db_health_data)
    db.commit()
    db.refresh(db_health_data)
    return db_health_data

def get_user_health_data(db: Session, user_id: int):
    return db.query(models.HealthData).filter(models.HealthData.user_id == user_id).all()

def create_ai_analysis(db: Session, analysis: schemas.AIAnalysisCreate):
    db_analysis = models.AIAnalysis(**analysis.dict())
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def get_user_ai_analyses(db: Session, user_id: int):
    return db.query(models.AIAnalysis).filter(models.AIAnalysis.user_id == user_id).all()

def create_doctor_review(db: Session, review: schemas.DoctorReviewCreate):
    db_review = models.DoctorReview(**review.dict())
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

def get_doctor_reviews(db: Session, doctor_id: int):
    return db.query(models.DoctorReview).filter(models.DoctorReview.doctor_id == doctor_id).all()

def get_analysis_reviews(db: Session, analysis_id: int):
    return db.query(models.DoctorReview).filter(models.DoctorReview.analysis_id == analysis_id).all() 