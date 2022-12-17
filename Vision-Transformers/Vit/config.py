import numpy as np
import wget
import torch

class Base:
    img_size : int = 244
    patch_size : int = 16
    layers : int = 12
    d_size : int = 768
    mlp_size : int = 3072
    n_heads : int = 12
    p_att : float = 0.1
    p_emb : float = 0.1
    p_mlp : float = 0.1
    out_channels : int = 21843
    eps : float = 1e-6
    pre_logits = True

def get_config(config_name, pretrained, fine_tuned):
    url = ""
    base = Base()
    if config_name == "B_16":
        if pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"

        elif fine_tuned and pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz"
            base.classes = 1000

    elif config_name == "B_32":
        base.patch_size = 32
        if pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz"

        elif fine_tuned and pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_32.npz"
            base.classes = 1000

    elif config_name == "L_16":
        base.layers = 24
        base.d_size = 1024
        base.mlp_size = 4096
        base.heads = 16

        if pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz"

        elif fine_tuned and pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16.npz"
            base.classes = 1000
    
    elif config_name == "L_32":
        base.layers = 24
        base.d_size = 1024
        base.mlp_size = 4096
        base.heads = 16
        base.patch_size = 32

        if pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz"

        elif fine_tuned and pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_32.npz"
            base.classes = 1000

    return base, url 

def get_weights(config_name, pretrained, fine_tuned):
    config, url = get_config(config_name, pretrained, fine_tuned)

    if url != "":
        filename = wget.download(url, out="weights")
        npy_weights = np.load(filename) # numpy weights
        pre_logits, torch_weights = rename_weights(npy_weights) # convert numpy to torch weights and rename them

    else:
        pre_logits = False
        torch_weights = 0

    return config, pre_logits, torch_weights

def load_weights(torch_weights, model):
    matches = []
    for name, params in torch_weights.items():
        # Checking if the name of the weight exists in model 
        if name in model.state_dict():
            # Checking if shapes are correct
            if params.shape == model.state_dict()[name].shape :
                matches.append(True)
    
    # Checking if all shapes are correct
    assert all(matches), " Some of the weights are different than in the original model. "
    torch.save(torch_weights, "model_weights.pth")
    model.load_state_dict(torch.load("model_weight.pth"))
    return model

def rename_weights(npy_weights):
    torch_weights = {}
    fixed_w = {}
    pre_logits = False

    for name, weight in npy_weights.items():
        n = name.replace("/", ".")
        w = weight
        
        n = n.replace("Transformer", "")
        n = n.replace("encoderblock_", "vit_blocks.")

        n = n.replace("LayerNorm_0", "ln1")
        n = n.replace("LayerNorm_2", "ln2")

        n = n.replace("Dense_0", "ff.0")
        n = n.replace("Dense_1", "ff.2")

        n = n.replace("MultiHeadDotProductAttention_1", "mha")
        n = n.replace("MlpBlock_3", "mlp")

        n = n.replace("embedding", "embeddings.projection")
        n = n.replace("posembed_input.pos_embeddings.projection", "embeddings.positions")
        n = n.replace("cls", "embeddings.cls_token")

        n = n.replace("kernel", "weight")
        n = n.replace("scale", "weight")
        n = n.replace("out", "linear") 
        
        n = n.replace("pre_logits", "pre_logits_layer")

        if "query" in n:
            n = n.replace("query", "Q")        
            if "weight" in n :
                w = w.reshape(w.shape[0], w.shape[1] * w.shape[2])
            if "bias" in n:
                w = w.reshape(-1)

        elif "key" in n:
            n = n.replace("key", "K")
            if "weight" in n :
                w = w.reshape(w.shape[0], w.shape[1] * w.shape[2])
            if "bias" in n:
                w = w.reshape(-1)

        elif "value" in n:
            n = n.replace("value", "V")
            if "weight" in n :
                w = w.reshape(w.shape[0], w.shape[1] * w.shape[2])
            if "bias" in n:
                w = w.reshape(-1)

        elif "linear" in n:
            if "weight" in n:
                w = w.reshape(w.shape[0] * w.shape[1], w.shape[2])

        elif "ff.0" in n or "ff.2" in n:
            if "weight" in n:
                w = w.reshape(w.shape[1], w.shape[0])

        elif "embeddings.positions" in n:
            w = w.reshape(w.shape[0], w.shape[2], w.shape[1])

        elif "embeddings.cls_token" in n:
            w = w.reshape(w.shape[0], w.shape[2], w.shape[1])

        elif "head" in n:
            if "weight" in n:
                w = w.reshape(w.shape[1], w.shape[0])

        elif "embeddings.projection.weight" in n:
            if "weight" in n:
                p, _, c, d = w.shape
                w = w.reshape(d, c, p, p)

        if n[0] == ".":
            n = n[1:]

        if "pre_logits" in n:
            pre_logits = True

        torch_weights[n] = torch.from_numpy(w)
        print(n, w.shape)

    return pre_logits, torch_weights




