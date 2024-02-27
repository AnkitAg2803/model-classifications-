import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)

def plot_pairplot(dataset):
    sns.pairplot(dataset, hue='variety')
    plt.show()
def plot_boxplot(dataset):
    sns.boxplot(x='variety', y='sepal.length', data=dataset)  # Use the correct column name for x-axis
    plt.show()
    sns.boxplot(x='variety', y='sepal.width', data=dataset)   # Use the correct column name for x-axis
    plt.show()
    sns.boxplot(x='variety', y='petal.length', data=dataset)  # Use the correct column name for x-axis
    plt.show()
    sns.boxplot(x='variety', y='petal.width', data=dataset)   # Use the correct column name for x-axis
    plt.show()


if __name__ == "__main__":
    dataset_path = input("Enter the path to your dataset: ")
    dataset = load_dataset(dataset_path)

    # Display pairplot
    plot_pairplot(dataset)

    # Display boxplot
    plot_boxplot(dataset)
