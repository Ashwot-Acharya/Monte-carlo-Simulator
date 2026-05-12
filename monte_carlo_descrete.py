import pandas as pd
import matplotlib.pyplot as plt
import random
import bisect


def read_data(filepath: str) -> list:
    df = pd.read_csv(filepath)
    return df['frequency'].tolist()


def random_number_generator(n: int) -> list:
    return [random.randint(0, 99) for _ in range(n)]


def frequency_distribution(freq_list: list) -> list:
    total = sum(freq_list)
    return [x / total for x in freq_list]


def assign_random(filepath: str, newfile: str, distribution: list) -> None:
    df = pd.read_csv(filepath)
    col = df['x_val'].tolist()

    assert len(col) >= len(distribution), "Not enough x_val rows for distribution"

    start_tags, end_tags = [], []
    s = 0
    for x in distribution:
        e = s + round(x * 1000) - 1  # round instead of int to reduce drift
        start_tags.append(s)
        end_tags.append(e)
        s = e + 1

    pd.DataFrame({
        'x_val': col[:len(start_tags)],
        'start_tag': start_tags,
        'end_tag': end_tags
    }).to_csv(newfile, index=False)


def simulate(random_nums: list, generated_csv: str) -> list:
    df = pd.read_csv(generated_csv)
    x_vals = df['x_val'].tolist()
    start_tags = df['start_tag'].tolist()

    recorded = []
    for y in random_nums:
        # bisect finds the right bucket in O(log n) instead of O(n)
        idx = bisect.bisect_right(start_tags, y) - 1
        if idx >= 0 and idx < len(x_vals):
            recorded.append(x_vals[idx])
        else:
            recorded.append(None)  # explicit instead of silent drop
    return recorded


def plot_data(random_nums: list, recorded_data: list) -> None:
    plt.figure()
    plt.scatter(random_nums, recorded_data, alpha=0.6)
    plt.xlabel("Random Numbers (0–99)")
    plt.ylabel("Simulated Outcome")
    plt.title("Monte Carlo Simulation Results")
    plt.tight_layout()
    plt.savefig("plotted.png")
    print("Plot saved to plotted.png")


def main():
    frequency = read_data('./data.csv')
    random_numbers = random_number_generator(50)
    distribution = frequency_distribution(frequency)
    assign_random('./data.csv', './generated.csv', distribution)
    recorded_data = simulate(random_numbers, './generated.csv')

    print("Random numbers:", random_numbers)
    print("Recorded data: ", recorded_data)
    plot_data(random_numbers, recorded_data)


if __name__ == "__main__":
    main()