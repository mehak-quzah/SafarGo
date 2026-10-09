
# SafarGo: Reinforcement Learning MVP

**Author:** Mehak Quzah  
**GitHub Repository:** [SafarGo](https://github.com/mehak-quzah/SafarGo)

## Project Overview
SafarGo is a ride-hailing startup operating in Rawalpindi and Islamabad. This project uses a Python simulation to test how Q-learning can help allocate idle vehicles to different zones.

## Objectives
- Increase completed rides.
- Reduce passenger waiting time.
- Reduce fuel consumption.
- Compare Q-learning with baseline strategies.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Streamlit (for web deployment)

## Methods
The simulation compares three strategies:
1. Random vehicle allocation
2. Sending vehicles to the busiest zone
3. Q-learning

## Performance Metrics
- Completed rides
- Average waiting time
- Fuel consumption

The results are saved in a CSV file, and graphs are generated to compare the strategies.

## Installation
Install the required libraries:

```bash
pip install numpy pandas matplotlib streamlit
```

## Run the Simulation
Run the Python simulation:

```bash
python safargo.py
```

Replace `safargo.py` with your actual Python filename if it is different.

## Deployment
The application can be deployed using Streamlit Community Cloud.

**GitHub Repository:** https://github.com/mehak-quzah/SafarGo

**Live Deployment Link:** Add your Streamlit app URL after deployment.

## Ethical Considerations
- Ensure fair vehicle distribution across all zones.
- Explain AI recommendations to build driver trust.
- Protect passenger and driver data.
- Test with 10% of the fleet before wider deployment.
- Keep manual dispatch available as a backup.

## Limitations
This is a simplified simulation. Its results may not represent real traffic, demand, waiting times, or fuel consumption. Further testing with real-world data is needed.

## Conclusion
This project demonstrates how Q-learning can support vehicle allocation while highlighting the trade-offs between completed rides, waiting time, and fuel consumption.