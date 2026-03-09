from pydantic import BaseModel, Field
from typing import Optional


class EmployeeInput(BaseModel):
    """Données d'entrée pour la prédiction ML"""

    # Informations personnelles
    Age: int = Field(..., ge=18, le=60)
    Gender: str
    MaritalStatus: str

    # Informations professionnelles
    Department: str
    JobRole: str
    JobLevel: int = Field(..., ge=1, le=5)
    BusinessTravel: str
    EducationField: str
    Education: int = Field(..., ge=1, le=5)

    # Rémunération
    MonthlyIncome: int
    PercentSalaryHike: int
    StockOptionLevel: int = Field(..., ge=0, le=3)
    DailyRate: int
    HourlyRate: int
    MonthlyRate: int

    # Engagement
    JobSatisfaction: int = Field(..., ge=1, le=4)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    JobInvolvement: int = Field(..., ge=1, le=4)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    PerformanceRating: int = Field(..., ge=1, le=4)

    # Activité
    OverTime: str
    DistanceFromHome: int
    NumCompaniesWorked: int
    TotalWorkingYears: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    TrainingTimesLastYear: int

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Gender": "Male",
                "MaritalStatus": "Single",
                "Department": "Sales",
                "JobRole": "Sales Executive",
                "JobLevel": 2,
                "BusinessTravel": "Travel_Frequently",
                "EducationField": "Marketing",
                "Education": 3,
                "MonthlyIncome": 3000,
                "PercentSalaryHike": 11,
                "StockOptionLevel": 0,
                "DailyRate": 500,
                "HourlyRate": 60,
                "MonthlyRate": 10000,
                "JobSatisfaction": 2,
                "EnvironmentSatisfaction": 2,
                "RelationshipSatisfaction": 2,
                "JobInvolvement": 2,
                "WorkLifeBalance": 2,
                "PerformanceRating": 3,
                "OverTime": "Yes",
                "DistanceFromHome": 15,
                "NumCompaniesWorked": 5,
                "TotalWorkingYears": 8,
                "YearsAtCompany": 2,
                "YearsInCurrentRole": 1,
                "YearsSinceLastPromotion": 2,
                "YearsWithCurrManager": 1,
                "TrainingTimesLastYear": 2
            }
        }


class RiskFactor(BaseModel):
    """Facteur de risque individuel"""
    feature: str
    shap_value: float
    impact: str


class PredictionResponse(BaseModel):
    """Réponse complète de la prédiction"""
    probability: float
    risk_level: str
    threshold_used: float
    top_risk_factors: list[RiskFactor]
    recommendation_factors: list[str]


class HealthResponse(BaseModel):
    """Réponse health check"""
    status: str
    model_version: str
    threshold: float
    accuracy: float
    auc_roc: float
    fp: int
    fn: int