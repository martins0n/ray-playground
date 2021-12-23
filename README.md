# Ray Cluster Playground

## Cluster

Start cluster:

```bash
make ray-local-cluster-start
```

## Dashboard  

http://localhost:8265

## Examples  

Optuna job:
```
python ray-jobs/optuna_task.py
```


## Requirements
- docker-compose
- `pip install -r ray-jobs/requirements.txt`