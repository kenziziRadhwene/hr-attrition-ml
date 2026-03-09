from fastapi import APIRouter, HTTPException
from app.schemas import EmployeeInput, PredictionResponse, HealthResponse, RiskFactor
from app.predictor import predictor

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Vérifie l'état de santé du modèle ML
    """
    try:
        health = predictor.get_health()
        return HealthResponse(**health)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict_attrition(employee: EmployeeInput):
    """
    Prédit le risque d'attrition d'un employé

    - **probability** : probabilité de départ (0 à 1)
    - **risk_level** : FAIBLE / MOYEN / ÉLEVÉ
    - **top_risk_factors** : facteurs SHAP principaux
    - **recommendation_factors** : actions préventives recommandées
    """
    try:
        # Conversion en dictionnaire
        employee_data = employee.model_dump()

        # Prédiction
        result = predictor.predict(employee_data)

        # Conversion des facteurs SHAP
        risk_factors = [
            RiskFactor(
                feature=f['feature'],
                shap_value=f['shap_value'],
                impact=f['impact']
            )
            for f in result['top_risk_factors']
        ]

        return PredictionResponse(
            probability=result['probability'],
            risk_level=result['risk_level'],
            threshold_used=result['threshold_used'],
            top_risk_factors=risk_factors,
            recommendation_factors=result['recommendation_factors']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(employees: list[EmployeeInput]):
    """
    Prédit le risque d'attrition pour plusieurs employés
    """
    try:
        results = []
        for employee in employees:
            employee_data = employee.model_dump()
            result = predictor.predict(employee_data)

            risk_factors = [
                RiskFactor(
                    feature=f['feature'],
                    shap_value=f['shap_value'],
                    impact=f['impact']
                )
                for f in result['top_risk_factors']
            ]

            results.append(PredictionResponse(
                probability=result['probability'],
                risk_level=result['risk_level'],
                threshold_used=result['threshold_used'],
                top_risk_factors=risk_factors,
                recommendation_factors=result['recommendation_factors']
            ))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))