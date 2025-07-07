import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

def initialization(N, dim, lb, ub):
    return np.random.rand(N, dim) * (ub - lb) + lb

def intensity(N, Xprey, X):
    I = np.zeros(N)
    for i in range(N):
        di = np.linalg.norm(X[i] - Xprey + np.finfo(float).eps)**2
        if i < N - 1:
            S = np.linalg.norm(X[i] - X[i + 1] + np.finfo(float).eps)**2
        else:
            S = np.linalg.norm(X[i] - X[0] + np.finfo(float).eps)**2
        r2 = np.random.rand()
        I[i] = r2 * S / (4 * np.pi * di)
    return I

def evaluate_fitness(X, X_data, y_data):
    fitness = []
    for individual in X:
        selected_features = np.where(individual > 0.5)[0]
        if len(selected_features) == 0:
            fitness.append(1.0)  # Worst possible score
            continue
        clf = DecisionTreeClassifier()
        X_subset = X_data[:, selected_features]
        score = cross_val_score(clf, X_subset, y_data, cv=5).mean()
        fitness.append(1 - score)
    return np.array(fitness)

def HBA_DTC(X_data, y_data, dim, lb, ub, tmax, N):
    beta = 6
    C = 2
    vec_flag = [1, -1]

    X = initialization(N, dim, lb, ub)
    fitness = evaluate_fitness(X, X_data, y_data)
    GYbest = np.min(fitness)
    gbest = np.argmin(fitness)
    Xprey = X[gbest].copy()
    CNVG = []

    for t in range(tmax):
        alpha = C * np.exp(-t / tmax)
        I = intensity(N, Xprey, X)
        Xnew = np.copy(X)
        for i in range(N):
            r = np.random.rand()
            F = vec_flag[np.random.randint(2)]
            for j in range(dim):
                di = Xprey[j] - X[i, j]
                if r < 0.5:
                    r3, r4, r5 = np.random.rand(3)
                    Xnew[i, j] = (Xprey[j] + F * beta * I[i] * Xprey[j] +
                                  F * r3 * alpha * di * abs(np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5))))
                else:
                    r7 = np.random.rand()
                    Xnew[i, j] = Xprey[j] + F * r7 * alpha * di

            Xnew[i] = np.clip(Xnew[i], lb, ub)

        new_fitness = evaluate_fitness(Xnew, X_data, y_data)
        for i in range(N):
            if new_fitness[i] < fitness[i]:
                fitness[i] = new_fitness[i]
                X[i] = Xnew[i]

        Ybest = np.min(fitness)
        CNVG.append(Ybest)
        if Ybest < GYbest:
            GYbest = Ybest
            Xprey = X[np.argmin(fitness)]

    return Xprey, 1 - GYbest, CNVG

# Example run with Iris dataset
if __name__ == "__main__":
    iris = load_iris()
    X_data = iris.data
    y_data = iris.target
    best_solution, best_score, convergence = HBA_DTC(X_data, y_data, dim=X_data.shape[1],
                                                     lb=0, ub=1, tmax=50, N=10)
    print("Best Feature Mask:", best_solution)
    print("Best Accuracy:", best_score)
