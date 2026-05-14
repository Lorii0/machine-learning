import percettrone
FEATURES = 5
THRESHOLD = 0.5

def main():
    weights = []
    inputs = []
    question = ["Artista famoso? (0 o 1):", "Bel meteo? (0 o 1):", "Amici presenti? (0 o 1):", "Cibo buono? (0 o 1):", "Alcool disponibile? (0 o 1):"]
    bias = 0.0

    if not percettrone.load_data("data.txt", weights, bias):
        print("Error loading data")
        return

    print("Insert the data:\n")
    for i in range(FEATURES):
        print(question[i])
        a = int(input())
        if a != 0 and a != 1:
            print("Invalid input, please insert 0 or 1")
            return
        inputs.append(a)

    output = percettrone.predict(inputs, weights, bias)
    if output == 1:
        print("Esci stasera")
    else:
        print("Resta a casa")

if __name__ == "__main__":
    main()