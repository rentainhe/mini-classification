## saving and loading weights
### saving
#### 1. automatic saving
Lightning automatically saves the checkpoint with the state of last training epoch.

This makes sure you can resume training in case it was interrupted

`Change the checkpoint path pass` in
```
# save checkpoints to '/ckpts` at every epoch end
trainer = Trainer(default_root_dir='/ckpts')
```
- just use the `default_root_dir` args

#### 2. customize the checkpointing behavior
Example: update checkpoint based on `validation loss`
- step 1: calculate any metric or other quantity you with to monitor
- step 2: log the quantity use `log()` method, with a key such as `val_loss`
- step 3: initialize the `ModelCheckpoint` callback, and set monitor to be the key of your quantity
- step 4: Pass the callback to the callbacks `Trainer` flag 
```python
from pytorch_lightning.callbacks import ModelCheckpoint

class LitAutoEncoder(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        # 1. calculate loss
        loss = F.cross_entropy(y_hat, y)

        # 2. log `val_loss`
        self.log('val_loss', loss)

# 3. Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(monitor='val_loss')

# 4. Add your callback to the callbacks list
trainer = Trainer(callbacks=[checkpoint_callback])
```

#### 3. more options in `ModelCheckpoint`
- `dirpath`: the path to save your weights
- `filename`: the name of the file
- `save_top_k`: save the best `k` models according to the quantity
- `mode`: `"min"` or `"max"`, monitored quantity's min or max

example:
```python
from pytorch_lightning.callbacks import ModelCheckpoint

class LitAutoEncoder(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='my/path/',
    filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

trainer = Trainer(callbacks=[checkpoint_callback])
```

#### 4. manual saving
You can manually save checkpoints and restore your model from the checkpointed state
```python
model = MyLightningModule(hparams)
trainer.fit(model)
trainer.save_checkpoint("example.ckpt")
new_model = MyModel.load_from_checkpoint(checkpoint_path="example.ckpt")
```

### loading
#### 1. load weights, biases and hyperparameters
```python
model = MyLightingModule.load_from_checkpoint(PATH)

print(model.learning_rate)
# prints the learning_rate you used in this checkpoint

model.eval()
y_hat = model(x)
```

### restoring training state
#### 1. not just load weights but instead restore the full training
```python
model = LitModel()
trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

# automatically restores model, epoch, step, LR schedulers, apex, etc...
trainer.fit(model)
```
This will load all of the last train information, including `epoch`, `step`, `LR schedulers`

This means if you load the `epoch=85.ckpt` and want to train it about more `5` epochs, you should set `max_epoch=90`
```
trainer = Trainer(max_epochs=90, resume_from_checkpoint='./checkpoint_path/epoch=85.ckpt')
```


## Reference
- [save and load docs](https://pytorch-lightning.readthedocs.io/en/latest/weights_loading.html#checkpoint-loading)
- [Docs](https://pytorch-lightning.readthedocs.io/en/latest/weights_loading.html?highlight=saving)