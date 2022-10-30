import matplotlib.pyplot as plt
import pandas as pd

def plot_columns_of_df(df):
    for column in df.columns:
        plt.subplot(1, 2, 1)
        plt.title('{}'.format(column))
        plt.hist(df[column], 50, density=True, facecolor='g', alpha=0.75)
        plt.subplot(1, 2, 2)
        plt.boxplot(df[column])
        plt.legend([(key, round(val, 2)) for key, val in df[column].describe().iteritems()])
        plt.show()

if __name__ == "__main__":
    pass