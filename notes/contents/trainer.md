## Trainer
### `val_check_interval`
#### usage
```
Trainer(..., val_check_interval=0.25)
``` 
- `val_check_interval`: default = 1

- set `val_check_interval=0.25` means validate `4 times per epoch`

- set `val_check_interval=100` means validate `per 100 batch` 