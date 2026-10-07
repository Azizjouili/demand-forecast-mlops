# Kubernetes deployment (local kind cluster)

Deploy the MLflow server, run a training Job, and serve the API on a local
[kind](https://kind.sigs.k8s.io/) cluster.

## Prerequisites
- Docker Desktop running
- `kind` and `kubectl` installed
- The app image built and loaded into the cluster:
```bash
  docker build -t demand-forecast-mlops:local .
  kind create cluster --name demand --image kindest/node:v1.31.2
  kind load docker-image demand-forecast-mlops:local --name demand
```

## Deploy
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/mlflow.yaml
kubectl wait --for=condition=available deploy/mlflow -n demand --timeout=180s

# Train + register the first model into the in-cluster MLflow
kubectl apply -f k8s/train-job.yaml
kubectl logs -f job/train -n demand

# Serve
kubectl apply -f k8s/api.yaml
kubectl wait --for=condition=available deploy/api -n demand --timeout=120s
```

## Access
```bash
kubectl port-forward svc/api 8000:8000 -n demand      # http://localhost:8000
kubectl port-forward svc/mlflow 5000:5000 -n demand   # http://localhost:5000
```

## Tear down
```bash
kind delete cluster --name demand
```