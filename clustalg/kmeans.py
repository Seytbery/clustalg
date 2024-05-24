import numpy as np
import random
import time
import cv2

import matplotlib.pyplot as plt

random.seed(7)
np.random.seed(7)

replay = 0
table = {}
def get_initial_centroids(X, k):
    """
    Функция выбирает k случайных точек данных из набора данных X, повторяющиеся точки удаляются и заменяются новыми точками
    таким образом, в результате мы получаем массив из k уникальных точек. Найденные точки могут использоваться в качестве начальных центроидов для алгоритма k means
    Args:
        X (numpy.ndarray) : массив точек набора данных, размер N:D
        k (int): количество центроидов

    Returns:
        (numpy.ndarray): массив из k уникальных начальных центроидов, размер K:D

    """
    number_of_samples = X.shape[0]
    sample_points_ids = random.sample(range(0, number_of_samples), k)

    centroids = [tuple(X[id]) for id in sample_points_ids]
    unique_centroids = list(set(centroids))

    number_of_unique_centroids = len(unique_centroids)

    while number_of_unique_centroids < k:
        new_sample_points_ids = random.sample(range(0, number_of_samples), k - number_of_unique_centroids)
        new_centroids = [tuple(X[id]) for id in new_sample_points_ids]
        unique_centroids = list(set(unique_centroids + new_centroids))

        number_of_unique_centroids = len(unique_centroids)

    return np.array(unique_centroids)


def get_euclidean_distance(A_matrix, B_matrix):
    """
    Функция вычисляет евклидово расстояние между матрицами A и B.
    Например, C[2,15] - это расстояние между точкой 2 из матрицы A (A[2]) и точкой 15 из матрицы B (B[15]).
    Args:
        A_matrix (numpy.ndarray): Размер матрицы N1:D
        B_matrix (numpy.ndarray): Размер матрицы N2:D

    Returns:
        numpy.ndarray: Размер матрицы N1:N2
    """

    A_square = np.reshape(np.sum(A_matrix * A_matrix, axis=1), (A_matrix.shape[0], 1))
    B_square = np.reshape(np.sum(B_matrix * B_matrix, axis=1), (1, B_matrix.shape[0]))
    AB = A_matrix @ B_matrix.T

    C = -2 * AB + B_square + A_square

    return np.sqrt(C)


def get_clusters(X, centroids, distance_mesuring_method):
    """
    Функция находит k центроидов и присваивает каждой из N точек массива X один центроид
    Args:
        X (numpy.ndarray): массив выборочных точек размером N:D
        centroids (numpy.ndarray): массив центроидов, размер K:D
        distance_mesuring_method (function): функция, принимающая 2 матрицы A (N1:D) и B (N2:D) и возвращающая расстояние
        между всеми точками из матрицы A и всеми точками из матрицы B, размер N1:N2

    Returns:
        dict {cluster_number: list_of_points_in_cluster}
    """

    k = centroids.shape[0]

    clusters = {}

    distance_matrix = distance_mesuring_method(X, centroids)

    closest_cluster_ids = np.argmin(distance_matrix, axis=1)

    for i in range(k):
        clusters[i] = []

    for i, cluster_id in enumerate(closest_cluster_ids):
        clusters[cluster_id].append(X[i])

    return clusters


def has_centroids_covered(previous_centroids, new_centroids, distance_mesuring_method, movement_threshold_delta):
    """
    Функция проверяет, переместился ли какой-либо из центроидов больше, чем MOVEMENT_THRESHOLD_DELTA, если нет, мы предполагаем, что центроиды были созданы
    Args:
        previous_centroids (numpy.ndarray): массив из k старых центроидов, размер K:D
        new_centroids (numpy.ndarray): массив из k новых центроидов, размер K:D
        distance_mesuring_method (function): функция принимает 2 матрицы A (N1:D) и B (N2:D) и возвращает расстояние
        movement_threshold_delta (float): пороговое значение, если центроиды перемещаются меньше, мы предполагаем, что алгоритм охватил


    Returns: (boolean) : Истинно, если центроиды покрыты, ложно, если нет
    """
    distances_between_old_and_new_centroids = distance_mesuring_method(previous_centroids, new_centroids)
    centroids_covered = np.max(distances_between_old_and_new_centroids.diagonal()) <= movement_threshold_delta

    return centroids_covered


def perform_k_means_algorithm(X, k, distance_mesuring_method, movement_threshold_delta=0):
    """
    Функция выполняет алгоритм k-средних значений для заданного набора данных, находит и возвращает k центроидов
    Args:
        X (numpy.ndarray) : массив точек набора данных, размер N:D
        distance_mesuring_method (function): функция принимает 2 матрицы A (N1:D) и B (N2:D) и возвращает расстояние
        между всеми точками из матрицы A и всеми точками из матрицы B размер N1:N2.
        k (int): количество центроидов
        movement_threshold_delta (float): пороговое значение, если центроиды перемещаются меньше, мы предполагаем, что алгоритм охватил

    Returns:
        (numpy.ndarray): массив из k центроидов, размер K:D
    """
    start_time = time.time()
    new_centroids = get_initial_centroids(X=X, k=k)

    centroids_covered = False

    global replay
    replay = 0
    while not centroids_covered:
        previous_centroids = new_centroids
        clusters = get_clusters(X, previous_centroids, distance_mesuring_method)

        new_centroids = np.array([np.mean(clusters[key], axis=0, dtype=X.dtype) for key in sorted(clusters.keys())])

        centroids_covered = has_centroids_covered(previous_centroids, new_centroids, distance_mesuring_method, movement_threshold_delta)

        replay += 1

    print(f"Время {time.time() - start_time}")
    print(f" Принадлежность: \n")
    global table
    for index in range(len(new_centroids)):
        table[index] = {
            "cnt": new_centroids[index],
            "cls": clusters[index]
        }
        print(f" центройд: \n {new_centroids[index]} \n кластеры: {clusters[index]} \n")

    return new_centroids



if __name__ == '__main__':

    rangeData = 10000

    dataSet = []
    datatmp = []
    for i in range(1, rangeData + 1):
 #       datatmp.append(i)
        datatmp.append(random.randrange(rangeData))
        if i % 2 == 0:
            dataSet.append(datatmp)
            datatmp = []

    X = np.array(dataSet)
    k = 20
    result = perform_k_means_algorithm(X, k, get_euclidean_distance)
    print(f'Центройды: \n {result}')
    print(f'Кол-во итераций: {replay}')
    print(f"Исходные данные: {dataSet}")

    x, y = X.T
    '''x = []
    y = []
    for item in dataSet:
        x.append(item[0])
        y.append(item[1])'''

r = lambda: random.randint(0,255)

for key in table:
    t, s = np.array(table[key]['cls']).T
    z,x = np.array(table[key]['cnt']).T
    plt.scatter(t, s, color=('#%02X%02X%02X' % (r(),r(),r())))
    plt.scatter(z,x, s= 200,color=('#%02X%02X%02X' % (r(), r(), r())))


plt.show()