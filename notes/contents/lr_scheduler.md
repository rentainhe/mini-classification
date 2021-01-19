## scheduler
### define it in `LightningModule.configure_optimizers()`

#### return options
#### 1. single optimizer
```python
class example(LightningModule):
    ...
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
```

#### 2. dictionary with `'optimizer'` key and `'lr_scheduler'` key
Use `WarmupCosineSchedule` as an example
```python
class example(LightningModule):
    ...
    def configure_optimizers(self):
        optimier = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        train_scheduler = {
            'scheduler': WarmupCosineSchedule(optimier, warmup_steps=1000, t_total=80000),
            'interval': 'step',
            'frequency': 1
        }
        return {'optimizer':optimier, 'lr_scheduler':train_scheduler}
```
use `lr_dict` to define scheduler
- `scheduler`: lr_schedule
- `interval`: two options: `step` and `epoch`
  - `step`: do `scheduler.step()` after each `step`
  - `epoch`: do `scheduler.step()` after each `epoch`
- `frequency`: default 1, set `frequency:2` do `scheduler.step()` after each `2` (epoch/step)

#### 3. monitor the learning rate
define a `LearningRateMonitor`
```python
from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='step') # update the log file at each step
trainer = Trainer(..., callbacks[lr_monitor])
trainer.fit()
```

## Reference
If you want to know more details, please see the [Docs](https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html?highlight=schedule#configure-optimizers)