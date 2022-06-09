import torch

class Base:
    d_model : int = 768
    vocab_size : int = 50_257
    window : int = 1024
    layers : int = 12
    p : float = 0.1
    heads : int = 12
    inner_state : int = 3072
    device : str = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size : int = 512

class Config(Base):
    def __init__(
        self,
        name : str
        ):
        if name == "small":
            Base.d_model = 768
            Base.layers = 12
        elif name == "mid":
            Base.d_model = 1024
            Base.layers = 24
        elif name == "large":
            Base.d_model = 1280
            Base.layers = 36
        elif name == "mega":
            Base.d_model = 1600
            Base.layers = 48
        else:
            print("Input correct config name.")

if __name__ == "__main__":
    c = Config("mega")
    print(c.layers, c.d_model)