import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your fine-tuned model
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-0125:oeg-fitness::BzWZgjwF"

class BallisticPredictor:
    def __init__(self):
        self.model = FINE_TUNED_MODEL
        self.client = client

    def build_prompt(self, data: dict) -> str:
        return (
            f"caliber: {data['caliber']}, bullet_weight: {data['bullet_weight']}, bullet_length: {data['bullet_length']}, "
            f"muzzle_velocity: {data['muzzle_velocity']}, ballistic_coefficient: {data['ballistic_coefficient']}, "
            f"barrel_length: {data['barrel_length']}, sight_height: {data['sight_height']}, twist_rate: {data['twist_rate']}, "
            f"temperature: {data['temperature']}, altitude: {data['altitude']}, humidity: {data['humidity']}, "
            f"pressure: {data['pressure']}, wind_speed: {data['wind_speed']}, distance_from_zero: {data['distance_from_zero']}"
        )

    def predict(self, input_data: dict):
        prompt = self.build_prompt(input_data)
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=500
        )

        content = response.choices[0].message.content

        try:
            drop_chart = []
            parts = content.split(", ")
            for part in parts:
                if ":" in part:
                    range_part, drop_part = part.split(": ")
                    range_yd = int(range_part.replace(" yards", ""))
                    drop_in = float(drop_part.replace(" inches", ""))
                    drop_chart.append({"range_yd": range_yd, "drop_in": drop_in})
            return drop_chart
        except Exception as e:
            print("❌ Error parsing model output:")
            print(content)
            raise e
