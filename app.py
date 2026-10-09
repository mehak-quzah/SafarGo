
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

st.set_page_config(page_title="SafarGo Dashboard", layout="wide")
st.title("SafarGo: Reinforcement Learning MVP")
st.write("Compare Q-learning with random and busiest-zone strategies.")
# Create images folder automatically
os.makedirs("images", exist_ok=True)

# ---------------------------------
# 1. SafarGo simulation settings
# ---------------------------------
np.random.seed(42)

NUM_ZONES = 40
NUM_VEHICLES = 300
EPISODES = 300
STEPS = 48  # 48 decisions per simulated episode

ALPHA = 0.5
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.05

# Simulated passenger demand in each zone
zone_popularity = np.random.uniform(0.5, 1.5, NUM_ZONES)

# Approximate distances between zones (fuel-cost units)
zone_positions = np.random.uniform(0, 10, (NUM_ZONES, 2))
distances = np.sqrt(
    ((zone_positions[:, None] - zone_positions[None, :]) ** 2).sum(axis=2)
)

# Q-table: state = (current zone, demand level)
# action = destination zone
q_table = np.zeros((NUM_ZONES, 3, NUM_ZONES))


def get_demand(step):
    """Generate changing, simulated demand across 40 zones."""
    if 12 <= step % 24 <= 18:
        peak_factor = 1.5
    else:
        peak_factor = 1.0

    demand = np.random.poisson(
        zone_popularity * peak_factor * 0.8
    )
    return demand


def get_demand_level(demand):
    """Convert total demand into a small state category."""
    total = demand.sum()

    if total < 20:
        return 0
    elif total < 35:
        return 1
    return 2


def run_episode(policy, episode, training=False):
    """Run one simulated operating episode."""
    vehicles = np.random.multinomial(
        NUM_VEHICLES,
        np.ones(NUM_ZONES) / NUM_ZONES
    )

    total_reward = 0
    completed_rides = 0
    total_waiting_time = 0
    total_fuel = 0
    total_requests = 0

    for step in range(STEPS):
        demand = get_demand(step)

        # Each four available vehicles provide one ride slot
        capacity = vehicles // 4

        served = np.minimum(demand, capacity)
        completed_rides += served.sum()
        total_requests += demand.sum()

        # Waiting-time proxy: base waiting plus unserved demand
        unserved = np.maximum(demand - capacity, 0)
        waiting = 3 + 0.2 * unserved.sum()
        total_waiting_time += waiting

        # Select a zone with spare vehicles
        spare = vehicles - np.maximum(1, demand)
        candidates = np.where(spare > 0)[0]

        if len(candidates) == 0:
            reward = -1
            total_reward += reward
            continue

        source = candidates[np.argmax(spare[candidates])]
        demand_level = get_demand_level(demand)

        # Choose the repositioning destination
        if policy == "q_learning":
            if training and np.random.random() < EPSILON:
                destination = np.random.randint(NUM_ZONES)
            else:
                destination = np.argmax(
                    q_table[source, demand_level]
                )

        elif policy == "busiest":
            # Baseline: send vehicle to the zone with
            # the greatest unmet demand
            destination = np.argmax(demand - capacity)

        else:
            # Random baseline
            destination = np.random.randint(NUM_ZONES)

        # Reposition one vehicle
        if destination != source:
            fuel_used = distances[source, destination]
            total_fuel += fuel_used

            old_capacity = capacity.copy()
            old_unserved = np.maximum(
                demand - old_capacity, 0
            ).sum()

            vehicles[source] -= 1
            vehicles[destination] += 1

            new_capacity = vehicles // 4
            new_unserved = np.maximum(
                demand - new_capacity, 0
            ).sum()

            # Reward encourages serving demand and saving fuel
            reward = (
                2 * (old_unserved - new_unserved)
                - 0.1 * fuel_used
            )
        else:
            reward = -0.1

        total_reward += reward

        # Update Q-table during training
        if training and policy == "q_learning":
            next_state = get_demand_level(
                get_demand(step + 1)
            )
            old_q = q_table[source, demand_level, destination]

            future_q = np.max(
                q_table[destination, next_state]
            )

            q_table[source, demand_level, destination] = (
                old_q
                + ALPHA * (
                    reward
                    + GAMMA * future_q
                    - old_q
                )
            )

    return {
        "Reward": total_reward,
        "Completed rides": completed_rides,
        "Average waiting time (min)": (
            total_waiting_time / STEPS
        ),
        "Fuel units": total_fuel,
        "Requests": total_requests
    }


# ---------------------------------
# 2. Train Q-learning
# ---------------------------------
training_rewards = []

for episode in range(EPISODES):
    # Gradually reduce exploration
    EPSILON = max(
        EPSILON_MIN,
        1.0 - episode / EPISODES
    )

    result = run_episode(
        "q_learning", episode, training=True
    )

    training_rewards.append(result["Reward"])

print("Q-learning training completed.")


# ---------------------------------
# 3. Compare all three systems
# ---------------------------------
evaluation_episodes = 50

results = {
    "Random baseline": [],
    "Busiest-zone baseline": [],
    "Q-learning": []
}

for episode in range(evaluation_episodes):
    results["Random baseline"].append(
        run_episode("random", episode)
    )

    results["Busiest-zone baseline"].append(
        run_episode("busiest", episode)
    )

    results["Q-learning"].append(
        run_episode("q_learning", episode)
    )


# Calculate average performance
summary = []

for system, runs in results.items():
    summary.append({
        "System": system,
        "Average reward": np.mean(
            [r["Reward"] for r in runs]
        ),
        "Average waiting time (min)": np.mean(
            [r["Average waiting time (min)"] for r in runs]
        ),
        "Completed rides": np.mean(
            [r["Completed rides"] for r in runs]
        ),
        "Fuel units": np.mean(
            [r["Fuel units"] for r in runs]
        )
    })

df = pd.DataFrame(summary)
st.subheader("Performance Comparison")
st.dataframe(df.round(2), use_container_width=True)

print("\nPERFORMANCE COMPARISON")
print(df.round(2).to_string(index=False))

df.to_csv("safargo_results.csv", index=False)


# ---------------------------------
# 4. Plot the reward curve
# ---------------------------------
plt.figure(figsize=(9, 5))

plt.plot(training_rewards, label="Episode reward")

# Smooth the curve to make the general trend easier to see
window = 10

if len(training_rewards) >= window:
    moving_average = np.convolve(
        training_rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.plot(
        range(window - 1, EPISODES),
        moving_average,
        label="10-episode moving average"
    )

plt.xlabel("Training episode")
plt.ylabel("Total reward")
plt.title("SafarGo Q-learning Reward Curve")
plt.legend()
plt.tight_layout()
st.pyplot(plt.gcf())
plt.savefig("images/safargo_reward_curve.png")
plt.close()


# ---------------------------------
# 5. Plot performance metrics
# ---------------------------------
metrics = [
    ("Average waiting time (min)",
     "Average Passenger Waiting Time"),

    ("Completed rides",
     "Completed Rides"),

    ("Fuel units",
     "Fuel Consumption")
]

for column, title in metrics:
    plt.figure(figsize=(8, 5))

    bars = plt.bar(df["System"], df[column])

    # Display actual values above bars
    plt.bar_label(bars, fmt="%.2f", padding=4)

    plt.title(title)
    plt.ylabel(column)
    plt.xticks(rotation=10)

    # Allow space above bars for labels
    plt.margins(y=0.15)
    plt.tight_layout()

    filename = column.lower().replace(" ", "_")
    filename = filename.replace("(", "").replace(")", "")

    st.pyplot(plt.gcf())
    plt.savefig("images/" + filename + ".png")
    plt.show()
    plt.close()

print("\nActual values used in the graphs:")
print(
    df[["System"] + [m[0] for m in metrics]]
    .round(2)
    .to_string(index=False)
)

print("\nFiles saved:")
print("- safargo_results.csv")
print("- images/safargo_reward_curve.png")
print("- Three metric charts in the images folder")