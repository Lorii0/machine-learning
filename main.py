import sys
import percettrone

FEATURES = 5


def run_concert_decision():
    try:
        weights, bias = percettrone.load_data("data.txt")
    except Exception as error:
        print(f"Error loading weights: {error}")
        return 1

    questions = [
        "Artista famoso? (0 o 1): ",
        "Bel meteo? (0 o 1): ",
        "Amici presenti? (0 o 1): ",
        "Cibo buono? (0 o 1): ",
        "Alcool disponibile? (0 o 1): ",
    ]

    inputs = []
    print("Inserisci i dati:")
    for question in questions:
        while True:
            try:
                value = int(input(question))
            except ValueError:
                print("Valore non valido. Inserisci 0 o 1.")
                continue
            if value in (0, 1):
                inputs.append(value)
                break
            print("Valore non valido. Inserisci 0 o 1.")

    output = percettrone.predict(inputs, weights, bias)
    if output == 1:
        print("\nVai al concerto!")
    else:
        print("\nResta a casa.")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        percettrone.train_three_input()
        return 0
    return run_concert_decision()


if __name__ == "__main__":  #eseguire con arg "train" per allenare i pesi
    main()
