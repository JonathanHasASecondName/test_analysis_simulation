import numpy as np
from sklearn.cluster import KMeans



def cluster(datax, datay, no_clusters):
    X = np.vstack((datax, datay))

    # Create a KMeans object with two clusters
    kmeans = KMeans(n_clusters=no_clusters)

    # Fit the KMeans object to the data
    kmeans.fit(X)

    # Get the labels for each data point
    labels = kmeans.labels_

    # Calculate the average x value for each cluster
    averages = []
    for i in range(2):
        cluster = X[labels == i]
        average = np.mean(cluster)
        averages.append(average)

    # Print the labels and average x values
    for i in range(2):
        print("Cluster {}: average x value = {}".format(i, averages[i]))

