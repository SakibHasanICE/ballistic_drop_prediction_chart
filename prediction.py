import os
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Use your fine-tuned model name from `check_status.py`
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-0125:oeg-fitness::BzWZgjwF"

def predict_drop_chart(user_input: str):
    messages = [{"role": "user", "content": user_input}]
    
    response = client.chat.completions.create(
        model=FINE_TUNED_MODEL,
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
    except (ValueError, IndexError) as e:
        print("❌ Error parsing model output:")
        print(content)
        print(f"Error details: {e}")
        return None

def plot_drop_chart(drop_chart):
    ranges = [point["range_yd"] for point in drop_chart]
    drops = [point["drop_in"] for point in drop_chart]

    plt.figure(figsize=(10, 6))
    plt.plot(ranges, drops, marker='o', linestyle='-', color='blue')
    plt.title("Predicted Ballistic Drop Chart")
    plt.xlabel("Range (yards)")
    plt.ylabel("Drop (inches)")
    plt.grid(True)
    
    # 🌟 NEW CODE: Add text labels for each point on the graph
    for i in range(len(ranges)):
        plt.text(
            ranges[i] + 5, # X coordinate with a small offset for readability
            drops[i] - 1,  # Y coordinate with a small offset for readability
            f"{drops[i]:.2f}", # The text label, formatted to 2 decimal places
            fontsize=8,
            ha='left' # Horizontal alignment
        )

    plt.tight_layout()
    plt.show()

def print_table_chart(drop_chart):
    """
    Prints a formatted table of the drop chart data.
    """
    print("\n--- Ballistic Drop Table ---")
    print("{:<12} | {:<12}".format("Range (yards)", "Drop (inches)"))
    print("-" * 27)
    for point in drop_chart:
        print("{:<12} | {:<12.2f}".format(point["range_yd"], point["drop_in"]))
    print("-" * 27)

if __name__ == "__main__":
    input_prompt = (
        "caliber: 0.200, bullet_weight: 368.0, bullet_length: 3.0, muzzle_velocity: 3700, "
        "ballistic_coefficient: 0.70, barrel_length: 50.0, sight_height: 4.0, twist_rate: 11.0, "
        "temperature: 70, altitude: 500, humidity: 30, pressure: 29.92, wind_speed: 5, distance_from_zero: 200"
    )

    chart = predict_drop_chart(input_prompt)
    if chart:
        # Print the data in JSON format
        print("--- Predicted Drop Chart (JSON) ---")
        print(json.dumps(chart, indent=2))
        
        # Print the data in a table format
        print_table_chart(chart)
        
        # Plot the data with values on the points
        plot_drop_chart(chart)