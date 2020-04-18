import numpy as np
import random
import matplotlib.pyplot as plt

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


def grouping(data, centres, style):
    L = len(data)
    clusters = np.zeros((L, 1))
    if style == 'Cosine':
        for i in range(L):
            max_dist = np.sum(data[i] * centres[0]) / (np.sqrt(np.sum(np.square(data[i]))) * np.sqrt(np.sum(np.square(centres[0]))))
            for c in range(len(centres)):
                dist = np.sum(data[i] * centres[c]) / (np.sqrt(np.sum(np.square(data[i]))) * np.sqrt(np.sum(np.square(centres[c]))))
                if dist > max_dist:
                    max_dist = dist
                    clusters[i] = c
    else:
        for i in range(L):
            min_dist = 0.
            if style == 'Euclidean':
                min_dist = np.sqrt(np.sum(np.square(data[i] - centres[0])))
            elif style == 'Manhattan':
                min_dist = np.sum(np.abs(data[i] - centres[0]))
            for c in range(len(centres)):
                dist = 0.
                if style == 'Euclidean':
                    dist = np.sqrt(np.sum(np.square(data[i] - centres[c])))
                elif style == 'Manhattan':
                    dist = np.sum(np.abs(data[i] - centres[c]))
                if dist < min_dist:
                    min_dist = dist
                    clusters[i] = c
    return clusters


def move(data, clusters, centres, style):
    new_centres = np.zeros((len(centres), len(centres[0])))
    error = 0.
    if style == 'Euclidean':
        count = np.zeros(len(centres))
        for i in range(len(data)):
            ite = int(clusters[i])
            new_centres[ite] += data[i]
            count[ite] += 1
        for i in range(len(centres)):
            new_centres[i] = new_centres[i] / count[i]
            error += np.sum(np.abs(new_centres[i] - centres[i]))
    elif style == 'Manhattan':
        k = len(centres)
        d = [[] for i in range(k)]
        for i in range(len(data)):
            d[int(clusters[i])].append(data[i])
        for i in range(k):
            a = np.array(d[i])
            L = len(a)
            new_centre = list()
            for j in range(300):
                c = a[:, j]
                c.reshape(1, L)
                c.sort()
                if L % 2:
                    mid = c[L//2]
                else:
                    mid = (c[L//2] + c[L//2 - 1]) / 2
                new_centre.append(mid)
            new_centres[i] = np.array(new_centre)
            error += np.sum(np.abs(new_centres[i] - centres[i]))
    elif style == 'Cosine':
        k = len(centres)
        d = [[] for i in range(k)]
        for i in range(len(data)):
            d[int(clusters[i])].append(data[i])
        for i in range(k):
            new_ite = 0
            cluster = np.array(d[i])
            max_cosine = -500.
            lc = len(cluster)
            cos = np.zeros((lc, lc))
            for a in range(lc):
                for b in range(a, lc):
                    cos[a][b] = cos[b][a] = np.sum(cluster[a] * cluster[b]) / (np.sqrt(np.sum(np.square(cluster))) * np.sqrt(np.sum(np.square(cluster[b]))))
            for a in range(lc):
                cosine = np.sum(cos[a])
                if cosine > max_cosine:
                    max_cosine = cosine
                    new_ite = a
            new_centres[i] = cluster[new_ite]
            error += np.sum(np.abs(new_centres[i] - centres[i]))
    return new_centres, error


def evaluate(clusters, labels):
    L = len(clusters)
    tp = fp = tn = fn = 0
    for i in range(L):
        for j in range(i + 1, L):
            if clusters[i] == clusters[j]:
                if labels[i] == labels[j]:
                    tp += 1
                else:
                    fp += 1
            else:
                if labels[i] == labels[j]:
                    fn += 1
                else:
                    tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F_score = 2 * precision * recall / (precision + recall)
    return precision, recall, F_score


def k_mean(data, labels, k, style):
    ite = random.sample(range(1, len(data)), k)
    centres = np.zeros((k, 300))
    for i in range(k):
        centres[i] = data[ite[i]]
    clusters = np.zeros((len(data), 1))
    error = 1.
    while error > eps:
        clusters = grouping(data, centres, style)
        centres, error = move(data, clusters, centres, style)
    return evaluate(clusters, labels)


def run(style, norm, file):
    data, labels = readAll()
    if norm:
        for i in range(len(data)):
            data[i] = data[i] / np.sum(np.square(data[i]))
    x = [i for i in range(1, 11)]
    precision = [0 for i in range(10)]
    recall = [0 for i in range(10)]
    F_score = [0 for i in range(10)]
    Cases = 10
    print("0%", end='')
    for case in range(Cases):
        for i in x:
            p, r, f = k_mean(data, labels, i, style)
            precision[i - 1] += p / Cases
            recall[i - 1] += r / Cases
            F_score[i - 1] += f / Cases
            print('\r', end='')
            print(str(case * 10 + i) + "%", end='')
    plt.figure()
    plt.xlabel("k")
    plt.ylabel("value")
    plt.plot(x, precision, label="precision")
    plt.plot(x, recall, label="recall")
    plt.plot(x, F_score, label="F_score")
    plt.legend()
    plt.savefig('./result/' + file + '.png')
    return


def main():
    print("For plotting smooth figures, all tasks will run 10 times and plot the average. "
          "That may need some time. Please wait for a while.")

    print("Running task 2 ...")
    run('Euclidean', False, 'task2')
    print(" Done.")

    print("Running task 3 ...")
    run('Euclidean', True, 'task3')
    print(" Done.")

    print("Running task 4 ...")
    run('Manhattan', False, 'task4')
    print(" Done.")

    print("Running task 5 ...")
    run('Manhattan', True, 'task5')
    print(" Done.")

    print("Running task 6 ...")
    run('Cosine', False, 'task6')
    print(" Done.")

    print("You can look over all the running result in the file named 'result'.")
    return


if __name__ == '__main__':
    main()
