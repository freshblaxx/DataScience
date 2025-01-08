import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
from numpy import ndarray, arange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.font_manager import FontProperties
from matplotlib.axes import Axes

CLASS_EVAL_METRICS = ["accuracy", "precision", "recall", "f1"]
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, gca, savefig, show
def run_NB(train_X: ndarray, train_Y: ndarray, test_X: ndarray, test_Y: ndarray, metric: str) -> dict[str, float]:
    model = GaussianNB()
    model.fit(train_X, train_Y)
    predictions = model.predict(test_X)
    return {
        "accuracy": accuracy_score(test_Y, predictions),
        "precision": precision_score(test_Y, predictions, average='weighted'),
        "recall": recall_score(test_Y, predictions, average='weighted'),
        "f1": f1_score(test_Y, predictions, average='weighted')
    }

def run_KNN(train_X: ndarray, train_Y: ndarray, test_X: ndarray, test_Y: ndarray, metric: str) -> dict[str, float]:
    model = KNeighborsClassifier()
    model.fit(train_X, train_Y)
    predictions = model.predict(test_X)
    return {
        "accuracy": accuracy_score(test_Y, predictions),
        "precision": precision_score(test_Y, predictions, average='weighted'),
        "recall": recall_score(test_Y, predictions, average='weighted'),
        "f1": f1_score(test_Y, predictions, average='weighted')
    }
def run_NB(train_X: ndarray, train_Y: ndarray, test_X: ndarray, test_Y: ndarray, metric: str) -> dict[str, float]:
    model = GaussianNB()
    model.fit(train_X, train_Y)
    predictions = model.predict(test_X)
    return {
        "accuracy": accuracy_score(test_Y, predictions),
        "precision": precision_score(test_Y, predictions, average='weighted'),
        "recall": recall_score(test_Y, predictions, average='weighted'),
        "f1": f1_score(test_Y, predictions, average='weighted')
    }


def evaluate_approaches(
    train1: DataFrame, test1: DataFrame, 
    train2: DataFrame, test2: DataFrame, 
    target: str = "CLASS", metric: str = "accuracy"
) -> dict[str, list]:
    eval: dict[str, list] = {metric: [] for metric in CLASS_EVAL_METRICS}

    # Process first dataset
    trnY1 = train1.pop(target).values
    trnX1: ndarray = train1.values
    tstY1 = test1.pop(target).values
    tstX1: ndarray = test1.values
    eval_NB1: dict[str, float] = run_NB(trnX1, trnY1, tstX1, tstY1, metric=metric)
    eval_KNN1: dict[str, float] = run_KNN(trnX1, trnY1, tstX1, tstY1, metric=metric)

    # Process second dataset
    trnY2 = train2.pop(target).values
    trnX2: ndarray = train2.values
    tstY2 = test2.pop(target).values
    tstX2: ndarray = test2.values
    eval_NB2: dict[str, float] = run_NB(trnX2, trnY2, tstX2, tstY2, metric=metric)
    eval_KNN2: dict[str, float] = run_KNN(trnX2, trnY2, tstX2, tstY2, metric=metric)

    # Combine results for each metric
    for met in CLASS_EVAL_METRICS:
        eval[met] = [
            eval_NB1.get(met, 0), eval_KNN1.get(met, 0),
            eval_NB2.get(met, 0), eval_KNN2.get(met, 0)
        ]

    return eval


def set_chart_labels(ax: Axes, title: str, xlabel: str, ylabel: str) -> Axes:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

FONT_TEXT = FontProperties()

def plot_multibar_chart_comparison(
    group_labels: list, yvalues: dict, 
    ax: Axes = None, title: str = "", xlabel: str = "", ylabel: str = "", percentage: bool = False
) -> Axes:
    if ax is None:
        ax = gca()
    ax = set_chart_labels(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    bar_labels = list(yvalues.keys())
    index = arange(len(group_labels))
    bar_width = 0.2  # Adjust for multiple comparisons
    ax.set_xticks(index + bar_width, labels=group_labels)

    for i, (label, bar_yvalues) in enumerate(yvalues.items()):
        values = ax.bar(
            index + i * bar_width,
            bar_yvalues,
            width=bar_width,
            label=label,
        )
        format_str = "%.2f" if percentage else "%.0f"
        ax.bar_label(values, fmt=format_str, fontproperties=FONT_TEXT)

    ax.legend(fontsize="xx-small")
    return ax


# Load datasets
train1 = read_csv("/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Data Balancing/class_arrests_SMOTE.csv")
test1 = read_csv("/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Eval_for_Preparation/Arrests_testing_data.csv")
train2 = read_csv("/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Data Balancing/class_ny_arrest_under.csv")
test2 = read_csv("/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Eval_for_Preparation/Arrests_testing_data.csv")

figure()
eval_results = evaluate_approaches(train1, test1, train2, test2, target="CLASS", metric="recall")
plot_multibar_chart_comparison(
    ["NB1", "KNN1", "NB2", "KNN2"],
    eval_results,
    title="Comparison of Balancing methods",
    xlabel="SMOTE vs Undersampling",
    ylabel="Scores",
    percentage=True
)
savefig("comparison_eval_balancing.png")
show()
