import joblib
import pandas as pd

from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = joblib.load("model-v1.joblib")

    # The arguments and types the model takes as input
    def predict(self,
                carat: float = Input(description="Carat of the diamond"),
                shape: str = Input(description="Shape of the diamond"),
                cut: str = Input(description="Cut of the diamond"),
                color: str = Input(description="Color of the diamond"),
                clarity: str = Input(description="Clarity of the diamond"),
                report: str = Input(description="Report of the diamond"),
                type: str = Input(description="Type of the diamond")
    ) -> float:
        sample = {
        'carat': carat,
        'shape': shape,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'report': report,
        'type': type,
        }

        data_point = pd.DataFrame([sample])

        """Run a single prediction on the model"""
        prediction = self.model.predict(data_point).tolist()[0]
        
        return prediction