FEATURES = 5
THRESHOLD = 0.5

def activation(x):
    if x > THRESHOLD:
        return 1
    else:
        return 0

def load_data(filename,weights,bias):
    try:
        with open(filename, 'r') as f:
            for i in range(FEATURES):
                weights.append(float(f.readline().split(" ")[2].strip()))
            bias = float(f.readline().split(" ")[1].strip())
            return 1
    except :
        return 0

def predict(inputs, weights, bias):
    sum = bias
    for i in range(FEATURES):
        sum += inputs[i] * weights[i]
    return activation(sum)
