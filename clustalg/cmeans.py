import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time

def c_means(X, k, m=2, max_iters=1000, tol=1e-4):
    start_time = time.time()
    n = len(X)
    # Инициализируем центроиды и матрицу принадлежности
    centroids = X[np.random.choice(range(n), k, replace=False)]
    U = np.zeros((n, k))
    for i in range(n):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        U[i, np.argmin(distances)] = 1
    # Основной цикл алгоритма
    iters = 0
    U_prev = None
    while iters < max_iters:
        iters += 1
        # Вычисляем новые центроиды
        for i in range(k):
            U_i = U[:, i]
            if np.sum(U_i) == 0:
                continue
            centroids[i] = np.sum(U_i[:, np.newaxis] * X, axis=0) / np.sum(U_i)
        # Вычисляем новую матрицу принадлежности
        for i in range(n):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            U_i = U[i, :]
            U_i[:] = (1 / (1 + (distances / m) ** 2)) ** (1 / (m - 1))
            U_i /= np.sum(U_i)
        # Проверим, изменилась ли матрица принадлежности
        if U_prev is not None and np.linalg.norm(U - U_prev) < tol:
            break
        U_prev = U.copy()

    print(f"Время {time.time() - start_time}")
    # Возвращаем кластеры и центроиды
    clusters = np.argmax(U, axis=1)

    return clusters, centroids, iters
if __name__ == "__main__":
    # Создаем искусственные данные
    X, y_true = make_blobs(n_samples=10000, centers=20, random_state=0)
    clusters, centroids, iter = c_means(X, k=20, m=1.5)
    print(f"Кол-во итераций{iter}")
    print(f"Кластеры {clusters}")
    print(f"Центройды {centroids}")
    # Выводим результат
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5)
    plt.show()
