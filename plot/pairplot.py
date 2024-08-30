import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sns.set_theme(style="ticks")

# df = sns.load_dataset("penguins")

filename = "../data/aerodynamics_data_1000.csv"
df = pd.read_csv(filename, index_col=0)

plot_kws = {"alpha": 0.2, "edgecolor": None}

pair_grid = sns.pairplot(df, vars=df.columns[0:5], plot_kws=plot_kws)
pair_grid.fig.subplots_adjust(top=0.95)
pair_grid.fig.suptitle("Pairplot of Inputs")

pair_grid = sns.pairplot(df, vars=df.columns[5:], plot_kws=plot_kws)
pair_grid.fig.subplots_adjust(top=0.95)
pair_grid.fig.suptitle("Pairplot of Outputs")

plt.show()
