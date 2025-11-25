import numpy as np
import pandas as pd

# ===========================================================
# Helper function to generate an Opower-style HER dataset
# ===========================================================
def generate_opower_dataset(
    dataset_id,
    n_households=10000,
    n_days=730,
    seed=123,
    base_usage_mu=20,
    base_usage_sigma=5,
    temp_amplitude=20,
    treatment_effect=-1.0,
    high_usage_multiplier=2.0,
    noise_sd=3.0
):

    rng = np.random.default_rng(seed)

    # Time setup
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    dates_array = dates.to_numpy()  # <-- crucial fix
    day_of_year = dates.dayofyear.to_numpy()

    # Temperature pattern (numpy array)
    temp = (
        60
        + temp_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        + rng.normal(0, 3, size=n_days)
    )

    # Household-level baseline usage
    alpha = rng.normal(base_usage_mu, base_usage_sigma, size=n_households)
    alpha = np.clip(alpha, 5, None)

    # Identify high-usage households
    high_threshold = np.quantile(alpha, 0.9)
    is_high_usage = (alpha >= high_threshold).astype(int)

    # Stratified random assignment by quartile
    quartiles = pd.qcut(alpha, 4, labels=False)
    treatment = np.zeros(n_households, dtype=int)
    for q in range(4):
        idx = np.where(quartiles == q)[0]
        treated_idx = rng.choice(idx, size=len(idx) // 2, replace=False)
        treatment[treated_idx] = 1

    # Treatment effect heterogeneity
    tau_i = np.where(is_high_usage == 1,
                     treatment_effect * high_usage_multiplier,
                     treatment_effect)

    # Time fixed effects (drift + noise)
    time_trend = np.linspace(0, 0.5, n_days)
    lambda_t = time_trend + rng.normal(0, 0.3, n_days)

    # Build panel
    HH, TT = np.meshgrid(np.arange(n_households), np.arange(n_days), indexing="ij")

    df = pd.DataFrame({
        "household_id": HH.ravel(order="C") + 1,
        "date": dates_array[TT].ravel(order="C"),      # <-- use numpy array
        "temp": temp[TT].ravel(order="C"),
        "alpha_i": alpha[HH].ravel(order="C"),
        "is_high_usage": is_high_usage[HH].ravel(order="C"),
        "treatment": treatment[HH].ravel(order="C"),
        "lambda_t": lambda_t[TT].ravel(order="C"),
    })

    # Random noise
    epsilon = rng.normal(0, noise_sd, size=df.shape[0])

    # Apply treatment effect
    tau_map = tau_i[df["household_id"].to_numpy() - 1]

    df["daily_kwh"] = (
        df["alpha_i"]
        + df["lambda_t"]
        + 0.35 * df["temp"]
        + df["treatment"] * tau_map
        + epsilon
    )

    df["daily_kwh"] = df["daily_kwh"].clip(lower=0.1)

    # Save CSV in same folder as script
    filename = f"opower_experiment_{dataset_id}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

    return df

# ===========================================================
# Generate 11 different Opower-style datasets
# ===========================================================
configs = [
    # change parameters across experiments
    {"n_households": 800, "treatment_effect": -0.8, "seed": 10},
    {"n_households": 1200, "treatment_effect": -1.1, "seed": 20},
    {"n_households": 1000, "treatment_effect": -0.6, "seed": 30, "noise_sd": 4},
    {"n_households": 1500, "treatment_effect": -1.5, "seed": 40, "temp_amplitude": 25},
    {"n_households": 900,  "treatment_effect": -0.9, "seed": 50},
    {"n_households": 1300, "treatment_effect": -1.3, "seed": 60, "high_usage_multiplier": 2.5},
    {"n_households": 700,  "treatment_effect": -0.7, "seed": 70, "noise_sd": 2},
    {"n_households": 1100, "treatment_effect": -1.0, "seed": 80, "base_usage_sigma": 7},
    {"n_households": 950,  "treatment_effect": -0.5, "seed": 90},
    {"n_households": 1050, "treatment_effect": -1.4, "seed": 100, "temp_amplitude": 30},
    {"n_households": 1250, "treatment_effect": -1.2, "seed": 110, "noise_sd": 5},
]

# Run all 11 datasets
for i, cfg in enumerate(configs, start=1):
    generate_opower_dataset(dataset_id=i, **cfg)

print("All 11 datasets generated!")
