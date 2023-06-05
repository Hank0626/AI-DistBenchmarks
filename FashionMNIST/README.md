## Single GPU on torch
```
python fashionmnist_cls_torch.py --lr 1e-3 --batch_size 64 --epochs 20
```

## DDP of torch
```
python -m torch.distributed.launch --use_env fashionmnist_cls_torch_ddp.py --lr 1e-3 --batch_size 64 --epochs 20
```

## Ray
```
python fashionmnist_cls_ray.py --lr 1e-3 --batch_size 64 --epochs 20
```

## PyTorch Lightning
```
python fashionmnist_cls_pl.py --lr 1e-3 --batch_size 64 --epochs 20
```