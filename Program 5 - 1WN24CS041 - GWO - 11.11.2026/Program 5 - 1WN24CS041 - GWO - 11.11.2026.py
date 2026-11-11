import numpy as np


def warehouse_cost(position):
    x, y = position
    stores = np.array([
        [2.0, 3.0],
        [8.0, 4.0],
        [5.0, 9.0],
        [1.0, 7.0],
    ])
    trips = np.array([10.0, 15.0, 8.0, 12.0])
    distances = np.sqrt((stores[:, 0] - x) ** 2 + (stores[:, 1] - y) ** 2)
    return np.sum(trips * distances)


def grey_wolf_optimizer(obj_func, dim, lb, ub, pack_size, max_iter):
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")

    beta_pos = np.zeros(dim)
    beta_score = float("inf")

    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    positions = np.random.uniform(lb, ub, (pack_size, dim))

    for t in range(max_iter):
        for i in range(pack_size):
            positions[i] = np.clip(positions[i], lb, ub)
            fitness = obj_func(positions[i])

            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i].copy()
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i].copy()

        a = 2 - t * (2 / max_iter)

        for i in range(pack_size):
            for j in range(dim):
                r1, r2 = np.random.random(), np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.random(), np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.random(), np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3.0

    return alpha_pos, alpha_score


if __name__ == "__main__":
    np.random.seed(42)
    dimensions = 2
    lower_bound = 0.0
    upper_bound = 10.0
    wolves = 25
    iterations = 200

    best_location, min_cost = grey_wolf_optimizer(
        warehouse_cost,
        dimensions,
        lower_bound,
        upper_bound,
        wolves,
        iterations,
    )

    print("Optimal Warehouse Coordinates (x, y):", np.round(best_location, 4))
    print("Minimum Daily Weighted Distance:", round(min_cost, 4))
