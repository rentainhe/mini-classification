## LightningModule
### Basic Function 
#### 1. save hyperparameters
the `LightningModule` has `save_hyperparameters` function, this is what I did in this repo
```python
class example(LightningModule):
    def __init__(self, hparams):
        self.save_hyperparameters(hparams)
        ...

# if hyperparameters is a class, you can use hyperparameters.__dict__
example(hyperparameters) # hyperparameters should be a dict in this case
```
more options is referred in the [Docs](https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html?highlight=save_hyperparameters#save-hyperparameters)


