import json
import random
from py_ballisticcalc import (
    Shot, Weapon, Ammo, DragModel, Atmo, Velocity,
    Distance, Temperature, TableG1, Calculator
)

# Distances to record drop at (yards)
ranges_yd = [50, 100, 150, 200, 250, 300, 400, 500]

# Common calibers list for random selection
calibers = [
    ".223 Remington",
    ".308 Winchester",
    ".30-06 Springfield",
    ".300 Winchester Magnum",
    ".243 Winchester",
    ".270 Winchester",
    "6.5 Creedmoor",
    "7mm Remington Magnum"
]

def generate_sample():
    # Pick a random caliber
    caliber = random.choice(calibers)

    # Generate random ballistic parameters
    bc = round(random.uniform(0.2, 0.6), 3)           # ballistic coefficient
    mv = round(random.uniform(2500, 3100), 1)         # muzzle velocity (fps)
    weight = round(random.uniform(150, 200), 1)       # bullet weight (grains)
    sight_height = round(random.uniform(1.3, 2.2), 2) # sight height (inches)
    temp = random.randint(40, 90)                      # ambient temp (F)
    altitude = random.randint(0, 5000)                 # altitude (feet)
    humidity = random.random()                          # humidity (0.0-1.0)
    pressure = round(random.uniform(28.0, 30.5), 2)   # pressure (inHg)
    zero = 100  # zero range in yards

    try:
        # Setup ballistic objects
        dm = DragModel(bc, TableG1)
        ammo = Ammo(dm, Velocity.FPS(mv), Temperature.Fahrenheit(temp))
        weapon = Weapon(sight_height=Distance.Inch(sight_height))
        atmo = Atmo(
            altitude=Distance.Foot(altitude),
            temperature=Temperature.Fahrenheit(temp),
            humidity=humidity,
            pressure=pressure
        )
        shot = Shot(weapon=weapon, ammo=ammo, atmo=atmo)

        calc = Calculator()
        calc.set_weapon_zero(shot, Distance.Yard(zero))

        drop_chart = []

        # Calculate trajectory up to max range
        max_range = max(ranges_yd)
        traj_data = calc.fire(shot, trajectory_range=Distance.Yard(max_range))
        df = traj_data.dataframe()

        # Extract drop at each range
        for r in ranges_yd:
            row = df.iloc[(df['distance'] - r).abs().argmin()]
            drop_in = row['target_drop']
            drop_chart.append({"range_yd": r, "drop_in": round(drop_in, 2)})

        # Compose prompt string with all parameters including caliber
        prompt = (
            f"caliber: {caliber}, bullet_weight: {weight}, muzzle_velocity: {mv}, ballistic_coefficient: {bc}, "
            f"sight_height: {sight_height}, temperature: {temp}, altitude: {altitude}, "
            f"humidity: {round(humidity * 100)}, pressure: {pressure}, distance_from_zero: {zero}"
        )

        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(drop_chart)}
            ]
        }

    except Exception as ex:
        print("[SKIP]", ex)
        return None


if __name__ == "__main__":
    samples = []
    target_samples = 1500
    print(f"Generating {target_samples} ballistic samples...")

    while len(samples) < target_samples:
        sample = generate_sample()
        if sample:
            samples.append(sample)
        if len(samples) % 100 == 0:
            print(f"Generated {len(samples)} samples")

    # Save all samples to JSON Lines file
    with open("ballistic_dataset_1500.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"[✓] Dataset saved as ballistic_dataset_1500.jsonl")
