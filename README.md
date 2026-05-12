# Monte Carlo Simulator

A Python-based Monte Carlo simulation engine that uses inverse transform sampling to generate outcomes from empirical frequency distributions. This tool helps analyze and visualize simulated data, making it useful for decision-making and risk analysis.

## Overview

The simulator reads frequency data from a CSV file, constructs a probability distribution, and generates simulated outcomes through statistical sampling. The results are visualized to show the behavior of the simulated system.

## How It Works

The core method uses inverse transform sampling:

1. Read frequency data from CSV file
2. Normalize frequencies into probabilities and build a cumulative distribution function (CDF)
3. Map the CDF onto bins for lookup
4. Generate uniform random values and map each to an outcome via its bin
5. Plot and analyze the simulated sequence

This approach is mathematically equivalent to Monte Carlo integration, where properties of a distribution are estimated by drawing samples and analyzing aggregate behavior.

## Project Structure

```
.
├── simulator.py               # Main continuous simulator with animated visualization
├── monte_carlo_descrete.py    # Discrete simulator with scatter plot visualization
├── data.csv                   # Input data: x values and their frequencies
├── generated.csv              # Intermediate file: x values with assigned bins
└── output/                    # Generated visualizations and results
    ├── gaussian_diagnostics.png
    └── gaussian_joint.png
```

## Input Data Format

The simulator expects a CSV file with two columns:

```csv
x_val,frequency
1,10
2,25
3,40
```

## Results

![Gaussian Diagnostics](output/gaussian_diagnostics.png)

![Gaussian Joint Distribution](output/gaussian_joint.png)

## Installation and Usage

### Requirements

- Python 3.x
- pandas
- matplotlib

### Running the Simulator

```bash
pip install pandas matplotlib
python simulator.py
```

The simulator generates visualizations and saves output to the `output/` directory.

## Use Cases

- Risk assessment and uncertainty analysis
- Financial modeling and forecasting
- Decision support where probability distributions are known
- Statistical validation of models against empirical data
