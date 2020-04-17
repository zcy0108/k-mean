import numpy as np
import random

eps = 1E-6


def read(file, mark):
    with open(file, 'r') as f:
        obs = f.readlines()
    data = np.zeros((len(obs), 300))
    labels = np.zeros((len(obs), 1))
    ite = 0
    for ob in obs:
        features = ob.split()
        labels[ite] = mark
        for i in range(1, len(features)):
            data[ite][i - 1] = features[i]
        ite += 1
    return data, labels


def readAll():
    data, labels = read("data/animals", 1.)
    x, y = read("data/countries", 2.)
    data = np.vstack((data, x))
    labels = np.vstack((labels, y))
    x, y = read("data/fruits", 3.)
    data = np.vstack((data, x))
    labels = np.vstack((labels, y))
    x, y = read("data/veggies", 4.)
    data = np.vstack((data, x))
    labels = np.vstack((labels, y))
    return data, labels


def grouping(data, centres):
    L = len(data)
    clusters = np.zeros((L, 1))
    for i in range(L):
        min_dist = np.sqrt(np.sum(np.square(data[i] - centres[0])))
        for c in range(len(centres)):
            dist = np.sqrt(np.sum(np.square(data[i] - centres[c])))
            if dist < min_dist:
                min_dist = dist
                clusters[i] = c
    return clusters


def move(data, clusters, centres):
    new_centres = np.zeros((len(centres), len(centres[0])))
    count = np.zeros(len(centres))
    for i in range(len(data)):
        new_centres[int(clusters[i])] += data[i]
        count[int(clusters[i])] += 1
    error = 0.
    for i in range(len(centres)):
        new_centres[i] = new_centres[i] / count[i]
        error += np.sqrt(np.sum(np.square(new_centres[i] - centres[i])))
    return new_centres, error / len(centres)


def k_mean(data, labels, k, style):
    ite = random.sample(range(1, len(data)), k)
    centres = np.zeros((k, 300))
    for i in range(k):
        centres[i] = data[ite[i]]
    clusters = np.zeros((len(data), 1))
    if style == 'Euclidean distances':
        error = 1.
        while error > eps:
            clusters = grouping(data, centres)
            centres, error = move(data, clusters, centres)
            print(error)
    return clusters


def main():
    data, labels = readAll()
    k_mean(data, labels, 10, 'Euclidean distances')
    return


if __name__ == '__main__':
    main()
