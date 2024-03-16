import torch

data_path = "../data/ExampleModelStealingPub.pt"
N = 100

def prepare_dataset():
    # TODO Popracuj nad data augmentation
    return torch.load(data_path)

# Przygotuj pierwsze 10 obrazków na state-of-art do sprawdzania noise
def prepare_state_of_art(dataset):
    # TODO
    pass

# Wysyłaj kolejne N obrazków po K razy, żeby zredukować szum
def send_data(dataset, K):
    # TODO
    pass

# Wielokrotnie wysyłaj pliki ze state-of-art żeby oszacować noise
def estimate_noise(state_of_art):
    # TODO
    pass

def data_steal():
    K = 1

    # Spreparuj dataset do wykradania
    dataset = prepare_dataset()

    # Przygotuj pierwsze 10 plikow na state-of-the-art

    # Wysyłaj kolejne 100 obrazków po K razy ze spreparowanego datasetu

    # Oszacuj wariancję na podstawie state-of-art i zaktualizuj K
