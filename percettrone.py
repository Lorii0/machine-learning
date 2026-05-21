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

def train_three_input():
    N = 8
    EPOCHS = 100
    LEARNING_RATE = 0.1

    inputs = [
        [0, 0, 0], [0, 0, 1],
        [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1],
        [1, 1, 0], [1, 1, 1],
    ]

    expected = []
    for x in inputs:
        sole, tempo, stanco = x
        expected.append(1 if ((tempo == 1 and stanco == 0) or (sole == 1 and stanco == 0)) else 0)

    weights = [0.0, 0.0, 0.0]
    bias = 0.0

    for epoch in range(EPOCHS):
        for i in range(N):
            total = bias
            for j in range(3):
                total += weights[j] * inputs[i][j]
            output = activation(total)
            error = expected[i] - output
            for j in range(3):
                weights[j] += LEARNING_RATE * error * inputs[i][j]
            bias += LEARNING_RATE * error

    print("Pesi allenati:")
    for i, w in enumerate(weights):
        print(f"Peso {i}: {w}")
    print(f"Bias: {bias}\n")
    print("Test del percettrone:")
    for i, x in enumerate(inputs):
        total = bias
        for j in range(3):
            total += weights[j] * x[j]
        output = activation(total)
        print(f"Input [{x[0]}, {x[1]}, {x[2]}] => Correre: {output} (Atteso: {expected[i]})")
