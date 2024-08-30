import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["text.usetex"] = True


filename = "data/aerodynamics_data_50000.csv"
df = pd.read_csv(filename, index_col=0)

for i, col in enumerate(df.columns[5:]):
    counts, bins = np.histogram(df[col].to_numpy(), bins=50)
    density = counts / sum(counts)
    fig, ax = plt.subplots()
    ax.hist(bins[:-1], bins, weights=density, edgecolor=ax.get_facecolor())
    ax.set_title(f"{col} Histogram of {len(df):,} Samples", loc="left")
    ax.set(xlabel=col, ylabel="Density $f_Y(y)$")
plt.show()
