from collections import Counter
from linear_algebra import distance
from statistics import mean
import math, random
import matplotlib.pyplot as plt
from knn import knn_classify
from data import cities


def plot_state_borders(plt, color='0.8'):
    pass


def plot_cities(cities):
    # key is language, value is pair (longitudes, latitudes)
    plots = {"Java": ([], []), "Python": ([], []), "R": ([], [])}

    # we want each language to have a different marker and color
    markers = {"Java": "o", "Python": "s", "R": "^"}
    colors = {"Java": "r", "Python": "b", "R": "g"}

    for (longitude, latitude), language in cities:
        plots[language][0].append(longitude)
        plots[language][1].append(latitude)

    # create a scatter series for each language
    for language, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[language], marker=markers[language],
                    label=language, zorder=10)

    plot_state_borders(plt)  # assume we have a function that does this

    plt.legend(loc=0)  # let matplotlib choose the location
    plt.axis([-130, -60, 20, 55])  # set the axes
    plt.title("Favorite Programming Languages")
    plt.show()


def classify_and_plot_grid(cities, k=1):
    """
    TODO
    Classify and plot for Python, Java, and R languages.
    """
    plots = {"Java": ([], []), "Python": ([], []), "R": ([], [])}
    markers = {"Java": "o", "Python": "s", "R": "^"}
    colors = {"Java": "r", "Python": "b", "R": "g"}

    # Predict preferred language for each city using knn_classify() from knn.py.
    # longitude range (-130, -60)
    # latitude in range (20, 55)
    # Save the coordinate of prediction result in plots variable.
    # TODO
    for c in cities:
        point = c[0]
        language = c[1]
        # Predict language
        predicated_language = knn_classify(k,
                                           [c for c in cities if c[0][0] != point[0] and c[0][1] != point[1]],
                                           point)

        plots[predicated_language][0].append(point[0])
        plots[predicated_language][1].append(point[1])


    # create a scatter series for each language
    # See above plot_cities() to plot your prediction.
    # create a scatter series for each language
    for language, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[language], marker=markers[language],
                    label=language, zorder=10)

    plot_state_borders(plt)  # assume we have a function that does this

    plt.legend(loc=0)  # let matplotlib choose the location
    plt.axis([-130, -60, 20, 55])  # set the axes
    plt.title("Favorite Programming Languages")
    plt.show()


if __name__ == "__main__":
    plot_cities(cities)
    classify_and_plot_grid(cities)
    classify_and_plot_grid(cities, 3)
    classify_and_plot_grid(cities, 5)
