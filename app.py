
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ---------------------------------
# Streamlit page settings
# ---------------------------------
st.set_page_config(
    page_title="SafarGo Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("SafarGo: Reinforcement Learning MVP")
st.write(
    "Compare Q-learning with random and busiest-zone strategies."
)

st.caption(
    "Simulation results are illustrative and do not represent real-world SafarGo data."
)

# Create images folder automatically
os.makedirs("images", exist_ok=True)


# ---------------------------------
# 1. SafarGo simulation settings
# ---------------------------------
np.random.seed(42)

NUM_ZONES = 40
NUM_VEHICLES = 300
EPISODES = 300
STEPS = 288  # 5-minute decisions for one simulated 24-hour day

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
    """Generate changing simulated demand across 40 zones."""
    # With 288 five-minute steps, steps 96-144 represent roughly 8 AM-noon.
    if 96 <= step <= 144:
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

        # Choose repositioning destination
        if policy == "q_learning":
            if training and np.random.random() < EPSILON:
                # Explore: try a random destination.
                destination = np.random.randint(NUM_ZONES)
            else:
                # Exploit: choose randomly among tied best actions.
                # This avoids always choosing zone 0 when Q-values are equal.
                state_q_values = q_table[source, demand_level]
                best_actions = np.flatnonzero(
                    state_q_values == state_q_values.max()
                )
                destination = int(np.random.choice(best_actions))

        elif policy == "busiest":
            destination = np.argmax(demand - capacity)

        else:
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

            # Reward encourages reducing unmet demand, moving toward
            # demand, and avoiding unnecessary travel.
            destination_shortage = max(
                0, int(demand[destination] - capacity[destination])
            )
            reward = (
                2.0 * (old_unserved - new_unserved)
                + 0.5 * min(int(demand[destination]), destination_shortage)
                - 0.1 * fuel_used
            )

        else:
            reward = -0.1

        total_reward += reward

        # Update Q-table during training
        if training and policy == "q_learning":
            # Use the observed demand level for this transition.
            # The next state is represented by the destination zone
            # and this simplified demand category.
            next_state = demand_level

            old_q = q_table[
                source, demand_level, destination
            ]

            future_q = np.max(
                q_table[destination, next_state]
            )

            q_table[
                source, demand_level, destination
            ] = (
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
st.subheader("Training Q-learning")

progress_bar = st.progress(0)
status_text = st.empty()

training_rewards = []

for episode in range(EPISODES):
    # Gradually reduce exploration
    EPSILON = max(
        EPSILON_MIN,
        1.0 - episode / EPISODES
    )

    result = run_episode(
        "q_learning",
        episode,
        training=True
    )

    training_rewards.append(result["Reward"])

    # Update progress occasionally
    if episode % 10 == 0 or episode == EPISODES - 1:
        progress_bar.progress(
            (episode + 1) / EPISODES
        )
        status_text.write(
            f"Training episode: {episode + 1}/{EPISODES}"
        )

progress_bar.empty()
status_text.success("Q-learning training completed.")


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

# Save the full-precision results
df.to_csv("safargo_results.csv", index=False)

# ---------------------------------
# 4. Display performance table
# ---------------------------------
st.divider()
st.subheader("Performance Comparison")

# Two decimal places to match the Word report
st.dataframe(
    df.round(2),
    use_container_width=True,
    hide_index=True
)

# Show key metrics
st.subheader("Key Results")

col1, col2, col3 = st.columns(3)

q_results = df[df["System"] == "Q-learning"].iloc[0]

with col1:
    st.metric(
        "Q-learning: Completed Rides",
        f'{q_results["Completed rides"]:.2f}'
    )

with col2:
    st.metric(
        "Q-learning: Waiting Time",
        f'{q_results["Average waiting time (min)"]:.2f} min'
    )

with col3:
    st.metric(
        "Q-learning: Fuel Units",
        f'{q_results["Fuel units"]:.2f}'
    )


# ---------------------------------
# 5. Plot the reward curve
# ---------------------------------
st.divider()
st.subheader("Q-learning Training Progress")

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)

ax.plot(
    training_rewards,
    label="Episode reward",
    linewidth=1.2,
    alpha=0.55
)

window = 10

if len(training_rewards) >= window:
    moving_average = np.convolve(
        training_rewards,
        np.ones(window) / window,
        mode="valid"
    )

    ax.plot(
        range(window - 1, EPISODES),
        moving_average,
        label="10-episode moving average",
        linewidth=2
    )

ax.set_xlabel("Training Episode")
ax.set_ylabel("Total Reward")
ax.set_title("SafarGo Q-learning Reward Curve")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    "images/safargo_reward_curve.png",
    dpi=150,
    bbox_inches="tight"
)

st.pyplot(fig, use_container_width=False)
plt.close(fig)


# ---------------------------------
# 6. Plot performance metrics
# ---------------------------------
st.divider()
st.subheader("Strategy Performance Comparison")

metrics = [
    (
        "Average waiting time (min)",
        "Average Passenger Waiting Time",
        "average_waiting_time"
    ),
    (
        "Completed rides",
        "Completed Rides",
        "completed_rides"
    ),
    (
        "Fuel units",
        "Fuel Consumption",
        "fuel_consumption"
    )
]

for column, title, filename in metrics:

    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=120)

    bars = ax.barh(
        df["System"],
        df[column]
    )

    # Display values at the ends of bars
    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=5,
        fontsize=9
    )

    ax.set_title(title)
    ax.set_xlabel(column)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # Leave space for labels
    max_value = df[column].max()

    if max_value > 0:
        ax.set_xlim(0, max_value * 1.22)

    fig.tight_layout()

    # Save only the graph image
    fig.savefig(
        f"images/{filename}.png",
        dpi=150,
        bbox_inches="tight"
    )

    # Display a compact graph in Streamlit
    st.pyplot(fig, use_container_width=False)

    plt.close(fig)


# ---------------------------------
# 7. Download results
# ---------------------------------
st.divider()
st.subheader("Download Simulation Results")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Results (CSV)",
    data=csv_data,
    file_name="safargo_results.csv",
    mime="text/csv"
)

st.caption(
    "Note: This is a simplified simulation. "
    "Waiting time and fuel consumption are estimates, "
    "not real-world measurements."
)

# ---------------------------------
# 8. Console output
# ---------------------------------
print("\nPERFORMANCE COMPARISON")
print(df.round(2).to_string(index=False))

print("\nActual values used in the graphs:")
print(
    df[["System"] + [m[0] for m in metrics]]
    .round(2)
    .to_string(index=False)
)

print("\nFiles saved:")
print("- safargo_results.csv")
print("- images/safargo_reward_curve.png")
print("- images/average_waiting_time.png")
print("- images/completed_rides.png")
print("- images/fuel_consumption.png")