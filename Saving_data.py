import joblib
import os

from Save_Best_Model import best_model
from Split_Data import scaler

os.makedirs("data", exist_ok=True)


# Save Encoders
joblib.dump(scaler, "data/scaler.pk1")
joblib.dump(best_model, "data/best_model.pk1")



