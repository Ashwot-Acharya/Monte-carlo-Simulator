
# Monte Carlo Simulator

A from-scratch Monte Carlo simulation engine in Python. Given an empirical frequency distribution from a CSV file, it performs **inverse transform sampling** to simulate outcomes and visualize the results — statically and animated.

---

## How It Works

The core method is **inverse transform sampling**:

1. Read frequency data from CSV
2. Normalize frequencies into probabilities → build a CDF
3. Map the CDF onto the integer range `[0, 1000]` as lookup bins
4. Generate uniform random integers, map each to an outcome via its bin
5. Plot the simulated sequence

```
Frequency CSV → Probability distribution → CDF → Bin assignment → Simulate → Plot
```

This is the same mathematical mechanism behind Monte Carlo integration: estimate properties of a distribution by drawing samples and aggregating.

---

## Files

```
.
├── simulator.py               # Continuous simulator with animated line plot
├── monte_carlo_descrete.py    # Discrete simulator with scatter plot
├── data.csv                   # Input: x_val and frequency columns
└── generated.csv              # Intermediate: x_val with assigned RNG bins
```

### CSV Format

```csv
x_val,frequency
1,10
2,25
3,40
```

---

## Usage

```bash
pip install pandas matplotlib
python simulator.py
```

Produces `plotted.png` and an animated plot window.

---

## Dependencies

- Python 3.x
- `pandas`
- `matplotlib`
