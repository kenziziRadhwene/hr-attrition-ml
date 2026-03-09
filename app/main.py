from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict_router import router

# ─────────────────────────────────────────
# Initialisation FastAPI
# ─────────────────────────────────────────
app = FastAPI(
    title="HR Attrition Predictor API",
    description="""
    ## API de prédiction du risque d'attrition RH

    Cette API utilise un modèle **XGBoost** entraîné sur le dataset IBM HR Analytics
    pour prédire le risque de départ d'un employé.

    ### Fonctionnalités :
    - **Prédiction individuelle** : score de risque pour un employé
    - **Prédiction batch** : score de risque pour plusieurs employés
    - **SHAP Values** : facteurs de risque explicables
    - **Recommandations** : actions préventives automatiques

    ### Niveaux de risque :
    - 🟢 **FAIBLE** : probabilité < seuil optimal
    - 🟡 **MOYEN** : probabilité entre seuil et 85%
    - 🔴 **ÉLEVÉ** : probabilité >= 85%
    """,
    version="2.0.0",
    contact={
        "name": "Ooredoo HR System",
        "email": "hr@ooredoo.com"
    }
)

# ─────────────────────────────────────────
# CORS — permet à Spring Boot d'appeler l'API
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Spring Boot
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
app.include_router(
    router,
    prefix="/api/ml",
    tags=["ML Predictions"]
)


# ─────────────────────────────────────────
# Root
# ─────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "HR Attrition Predictor API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/ml/health"
    }


# ─────────────────────────────────────────
# Lancement
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )