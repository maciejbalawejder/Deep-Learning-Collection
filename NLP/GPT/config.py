class Config:
    vocab_size : int = 5_000,
    window : int = 512,
    d_model = 768,
    layers : int = 12,
    p : float = 0.1,
    heads : int = 12,
    inner_state : int = 3072,
    device : str = "cpu"
