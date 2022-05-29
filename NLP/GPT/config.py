class Config:
    def __init__(
        self,
        vocab_size : int = 5_000,
        window : int = 512,
        d_model = 768,
        layers : int = 12,
        p : float = 0.1,
        heads : int = 12,
        inner_state : int = 3072,
        device : str = "cpu"
        ):

        self.vocab_size = vocab_size
        self.window = window
        self.layers = layers
        self.p = p
        self.heads = heads
        self.inner_state = inner_state
        self.d_model = d_model
        self.device = device