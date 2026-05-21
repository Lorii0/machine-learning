FEATURES = 5
THRESHOLD = 0.5

def activation(x):
    return 1 if x >= THRESHOLD else 0

def load_data(filename):
    weights = []
    with open(filename, 'r') as f:
        for i in range(FEATURES):
            line = f.readline()
            if not line:
                raise ValueError(f"Missing weight {i + 1} in {filename}")
            parts = line.strip().split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid weight line: {line.strip()}")
            weights.append(float(parts[1].strip()))
        line = f.readline()
        if not line:
            raise ValueError(f"Missing bias line in {filename}")
        parts = line.strip().split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid bias line: {line.strip()}")
        bias = float(parts[1].strip())
    return weights, bias

def predict(inputs, weights, bias):
    total = bias
    for i in range(FEATURES):
        total += inputs[i] * weights[i]
    return activation(total)

def _read_dataset(filename):
    X = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= FEATURES:
                xi = [int(p) for p in parts[:FEATURES]]
                X.append(xi)
    return X

def save_weights(filename, weights, bias):
    with open(filename, 'w') as f:
        for i, w in enumerate(weights, start=1):
            f.write(f"Peso {i}: {w}\n")
        f.write(f"Bias: {bias}\n")

def train_concert(epochs=100, learning_rate=0.1, save_file='data.txt'):
    X = _read_dataset('training_data.txt')
    Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    weights = [0.0 for _ in range(FEATURES)]
    bias = 0.0

    for epoch in range(epochs):
        for xi, yi in zip(X, Y):
            total = bias
            for j in range(FEATURES):
                total += weights[j] * xi[j]
            output = activation(total)
            error = yi - output
            for j in range(FEATURES):
                weights[j] += learning_rate * error * xi[j]
            bias += learning_rate * error

    print("Pesi allenati:")
    for i, w in enumerate(weights):
        print(f"Peso {i}: {w:.6f}")
    print(f"Bias: {bias:.6f}\n")

    print("Test del percettrone:")
    for xi, yi in zip(X, Y):
        total = bias
        for j in range(FEATURES):
            total += weights[j] * xi[j]
        out = activation(total)
        print(f"Input {xi} => Vai: {out} (Atteso: {yi})")

    save_weights(save_file, weights, bias)
    print(f"PesI salvati in: {save_file}")
    return weights, bias

