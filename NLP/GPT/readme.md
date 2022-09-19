# GPT
The Generative Pre-Training(GPT) by OpenAI was introducted in 2018 paper ["Improving Language Understanding
by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). The core idea was to pre-trained model on large corpus of unlabeled text in semi-supervised fashion, and then fine-tune the model for sepecific task.

![image](https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/GPT/images/training.png)

# Usage
```python
    from model import GPT
    from config import Config

    config = Config()
    gpt = GPT(config)
    lm_head = LMHead(config, gpt) # Language modeling head
    cls_head = CLSHead(config, gpt) # Classfication head
    logits = lm_head(torch.randint(0, config.vocab_size, (1, config.window)))
    lm_logits, cls_logits = cls_head(torch.randint(0, config.vocab_size, (1, config.window)))
    print(logits.shape)
    print(lm_logits.shape, cls_logits.shape)
    
```

# Architecture
Model largely follows the original transformer work but use decoder-only part. 


<p align="center">
<img 
  src="https://github.com/maciejbalawejder/DeepLearning-collection/blob/main/NLP/GPT/images/GPT-architecture.png"
>
</p>






