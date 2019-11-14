class Hparams:
    def __init__(self):
        """Experimental hyper parameters are here.

        All changes should be made here.

        """
        self.d = 200 # hidden size
        self.e = 125 # text embedding output dimension
        self.vocab = 32 # text vocab size
        self.F = 80 # mel spectrogram dimension
        self.g = 0.2 # attention parameter
        self.c = 512
        self.Fprime = 513