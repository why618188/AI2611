import numpy as np

class MKSVM:
    def __init__(self, kernels: list, penalty=3.0, max_iter=200, epsilon=1e-3, gamma=1.0, degree=3, bias=1.0):
        self.num_kernels = 0
        self.kernel_names = []
        valid_kernels = ['linear', 'polynomial', 'gaussian', 'sigmoid']
        for kernel in kernels:
            if kernel not in valid_kernels:
                raise ValueError(f"Invalid kernel. Expected one of {valid_kernels}, got {kernel}.")
            self.num_kernels += 1
            self.kernel_names.append(kernel)
        if self.num_kernels == 0:
            raise ValueError(f"No valid kernels.")

        self.penalty = penalty
        self.kernels = []
        self.kernel_weights = [1.0 / self.num_kernels] * self.num_kernels
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.gamma = gamma
        self.degree = degree
        self.bias = bias


    def compute_kernel(self, X, Z=None):
        kernels = []
        if Z is None:
            Z = X
        for i in range(self.num_kernels):
            kernel = self.compute_specific_kernel(self.kernel_names[i], X, Z)
            kernel = (kernel - np.mean(kernel)) / np.std(kernel)
            kernels.append(kernel)
        return kernels


    def compute_specific_kernel(self, kernel, X, Z):
        if kernel == 'linear':
            return np.dot(X, Z.T)
        elif kernel == 'polynomial':
            self.gamma = 0.02
            self.bias = 1.0
            return (self.gamma * np.dot(X, Z.T) + self.bias) ** self.degree
        elif kernel == 'gaussian':
            self.gamma = 0.0078
            X_square = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Z_square = np.sum(Z ** 2, axis=1).reshape(-1, 1)
            return np.exp(-self.gamma * (X_square + Z_square.T - 2 * np.dot(X, Z.T)))
        elif kernel == 'sigmoid':
            self.gamma = 0.001
            self.bias = -1.0
            return np.tanh(self.gamma * np.dot(X, Z.T) + self.bias)


    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.alphas = np.zeros(self.n_samples)
        self.X = X
        self.y = y
        self.b = 0

        self.kernels = self.compute_kernel(X)
        iteration = 0

        while iteration < self.max_iter:
            alpha_prev = np.copy(self.alphas)
            kernel = sum([kernel * weight for kernel, weight in zip(self.kernels, self.kernel_weights)])
            for i in range(self.n_samples):
                error_i = np.dot(self.alphas * self.y, kernel[:, i]) + self.b - self.y[i]

                # Check KKT Condition
                if not ((self.y[i] * error_i < -self.epsilon and self.alphas[i] < self.penalty) or
                        (self.y[i] * error_i > self.epsilon and self.alphas[i] > 0)):
                    continue

                # Select j
                candidates = [j for j in range(self.n_samples) if j != i]
                j = np.random.choice(candidates)
                error_j = np.dot(self.alphas * self.y, kernel[:, j]) + self.b - self.y[j]
                alpha_i, alpha_j = self.alphas[i], self.alphas[j]

                # Compute Bound
                if self.y[i] != self.y[j]:
                    lower = max(0., float(self.alphas[j] - self.alphas[i]))
                    upper = min(self.penalty, float(self.penalty + self.alphas[j] - self.alphas[i]))
                else:
                    lower = max(0., float(self.alphas[i] + self.alphas[j] - self.penalty))
                    upper = min(self.penalty, float(self.alphas[i] + self.alphas[j]))
                if lower == upper:
                    continue

                # Compute \eta
                eta = 2 * kernel[i, j] - kernel[i, i] - kernel[j, j]
                if eta >= 0:
                    continue

                # Update self.alphas
                new_alpha_j = np.clip(self.alphas[j] - self.y[j] * (error_i - error_j) / eta, lower, upper)
                if abs(new_alpha_j - alpha_j) < self.epsilon:
                    continue
                self.alphas[j] = new_alpha_j
                self.alphas[i] += self.y[i] * self.y[j] * (alpha_j - self.alphas[j])

                # Update b
                b1 = self.b - error_i - self.y[i] * (self.alphas[i] - alpha_i) * kernel[i, i] - self.y[j] * (
                        self.alphas[j] - alpha_j) * kernel[i, j]
                b2 = self.b - error_j - self.y[i] * (self.alphas[i] - alpha_i) * kernel[i, j] - self.y[j] * (
                        self.alphas[j] - alpha_j) * kernel[j, j]
                self.b = (b1 + b2) / 2

            # Update kernel weights
            vector = (self.alphas * self.y).reshape(-1, 1)
            g = [self.kernel_weights[i] ** 2 * (vector.T @ self.kernels[i] @ vector)[0, 0] for i in
                 range(self.num_kernels)]
            g = np.sqrt(np.array(g))
            self.kernel_weights = (g / np.sum(g)).tolist()

            if np.allclose(self.alphas, alpha_prev, atol=self.epsilon):
                break

            iteration += 1


    def predict(self, X):
        kernels = self.compute_kernel(X, self.X)
        kernel = sum([kernel * weight for kernel, weight in zip(kernels, self.kernel_weights)])
        decision_values = np.dot(kernel, self.alphas * self.y) + self.b
        return np.sign(decision_values)


    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


    def get_predict_value(self, X):
        kernels = self.compute_kernel(X, self.X)
        kernel = sum([kernel * weight for kernel, weight in zip(kernels, self.kernel_weights)])
        decision_values = np.dot(kernel, self.alphas * self.y) + self.b
        return decision_values

