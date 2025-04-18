from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deeprage.core import RageReport
import pandas as pd
import shap

app = FastAPI(title="DeepRage API", description="Natural-language queries over your datasets.")

class Query(BaseModel):
    dataset_path: str
    question: str

@app.post("/ask")
async def ask(q: Query):
    # Load dataset
    try:
        df = pd.read_csv(q.dataset_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot load dataset: {e}")

    # Initialize report
    rr = RageReport(df)
    text = q.question.lower()

    # Simple intent detection
    if "missing" in text or "null" in text:
        # Expect format: "missing train.csv test.csv target"
        raise HTTPException(status_code=422, detail="For missing-summary use the CLI or notebook magic.")
    elif "feature" in text or "important" in text:
        # Run SHAP feature importance
        model_info = rr.clean().propose_model(q.dataset_path)
        model = rr.propose_model(q.dataset_path)["model"]
        explainer = shap.Explainer(model, rr.df.drop(columns=[q.dataset_path]))
        shap_vals = explainer(rr.df.drop(columns=[q.dataset_path]))
        means = list(zip(rr.df.drop(columns=[q.dataset_path]).columns, shap_vals.values.mean(0)))
        top5 = sorted(means, key=lambda x: x[1], reverse=True)[:5]
        return {"top_features": top5}
    else:
        # Default: return a data preview
        sample = rr.df.head().to_dict(orient="records")
        return {"preview": sample}
