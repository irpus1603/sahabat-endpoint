# OpenShift Deployment Guide

This guide will help you deploy the Sahabat-9B API endpoint on OpenShift (OCP).

## Prerequisites

1. OpenShift CLI (`oc`) installed and configured
2. Access to an OpenShift cluster
3. Docker or Podman installed for building images
4. HuggingFace token (if needed for model access)

## Quick Start

### 1. Login to OpenShift

```bash
oc login <your-openshift-cluster-url>
```

### 2. Create a New Project (optional)

```bash
oc new-project sahabat-endpoint
```

Or use an existing project:

```bash
oc project <your-project-name>
```

### 3. Build and Push Docker Image

#### Option A: Using OpenShift BuildConfig

```bash
# Create a new build configuration
oc new-build --name=sahabat-endpoint \
  --binary=true \
  --strategy=docker

# Start the build from the current directory
oc start-build sahabat-endpoint --from-dir=. --follow

# Get the image stream
oc get is sahabat-endpoint
```

#### Option B: Using External Registry

```bash
# Build the image
docker build -t <registry-url>/sahabat-endpoint:latest .

# Push to registry
docker push <registry-url>/sahabat-endpoint:latest

# Update the image in openshift-deployment.yaml
# Replace 'sahabat-endpoint:latest' with your registry path
```

### 4. Create Secrets (if using HuggingFace token)

```bash
# Edit openshift-secrets.yaml and add your HuggingFace token
# Then apply:
oc apply -f openshift-secrets.yaml
```

Or create secret from command line:

```bash
oc create secret generic sahabat-secrets \
  --from-literal=huggingface-token=<your-token>
```

### 5. Deploy the Application

```bash
# Apply all configurations
oc apply -f openshift-deployment.yaml
oc apply -f openshift-service.yaml
oc apply -f openshift-route.yaml
```

Or apply all at once:

```bash
oc apply -f openshift-deployment.yaml,openshift-service.yaml,openshift-route.yaml
```

### 6. Verify Deployment

```bash
# Check deployment status
oc get deployments

# Check pods
oc get pods

# Check pod logs
oc logs -f deployment/sahabat-endpoint

# Check service
oc get svc sahabat-endpoint

# Get the route URL
oc get route sahabat-endpoint
```

### 7. Test the Deployment

```bash
# Get the route URL
ROUTE_URL=$(oc get route sahabat-endpoint -o jsonpath='{.spec.host}')

# Test health endpoint
curl https://${ROUTE_URL}/health

# Test chat completions
curl -X POST https://${ROUTE_URL}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

## Configuration

### Environment Variables

All environment variables are defined in `openshift-deployment.yaml`. Key configurations:

- **DEVICE**: Set to `cpu` or `cuda` (for GPU nodes)
- **WORKERS**: Number of uvicorn workers (default: 1)
- **MODEL_NAME**: HuggingFace model to use
- **LOG_LEVEL**: Logging level (INFO, DEBUG, WARNING, ERROR)

### Resource Limits

Default resource limits in the deployment:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

Adjust based on your cluster capacity and workload requirements.

### GPU Support

If deploying on GPU nodes:

1. Update the deployment to request GPU resources:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

2. Set environment variables:
```yaml
- name: DEVICE
  value: "cuda"
- name: USE_FLASH_ATTENTION
  value: "true"
```

## Scaling

### Manual Scaling

```bash
# Scale to 3 replicas
oc scale deployment/sahabat-endpoint --replicas=3
```

### Horizontal Pod Autoscaler (HPA)

```bash
oc autoscale deployment/sahabat-endpoint \
  --min=1 --max=5 \
  --cpu-percent=80
```

## Troubleshooting

### Check Pod Status

```bash
oc get pods
oc describe pod <pod-name>
```

### View Logs

```bash
# Follow logs
oc logs -f deployment/sahabat-endpoint

# Get logs from specific pod
oc logs <pod-name>

# Get previous logs (if pod crashed)
oc logs <pod-name> --previous
```

### Common Issues

1. **Pod not starting**: Check resource limits and node capacity
   ```bash
   oc describe pod <pod-name>
   ```

2. **Model download issues**: Ensure HuggingFace token is correct
   ```bash
   oc get secret sahabat-secrets -o yaml
   ```

3. **Out of memory**: Increase memory limits in deployment
4. **Slow startup**: Model loading can take 1-2 minutes, increase `initialDelaySeconds` in probes

### Debug Pod

```bash
# Get a shell in the running pod
oc rsh deployment/sahabat-endpoint

# Run interactive debug pod
oc debug deployment/sahabat-endpoint
```

## Update Deployment

### Update Image

```bash
# If using BuildConfig
oc start-build sahabat-endpoint --from-dir=. --follow

# If using external registry
docker build -t <registry-url>/sahabat-endpoint:v2 .
docker push <registry-url>/sahabat-endpoint:v2
oc set image deployment/sahabat-endpoint \
  sahabat-endpoint=<registry-url>/sahabat-endpoint:v2
```

### Update Configuration

```bash
# Edit deployment
oc edit deployment sahabat-endpoint

# Or update YAML and reapply
oc apply -f openshift-deployment.yaml
```

## Cleanup

```bash
# Delete all resources
oc delete -f openshift-deployment.yaml,openshift-service.yaml,openshift-route.yaml
oc delete secret sahabat-secrets

# Delete the entire project (if created)
oc delete project sahabat-endpoint
```

## Monitoring

### Check Metrics

```bash
# CPU and Memory usage
oc adm top pods

# Detailed metrics
oc describe pod <pod-name>
```

### Application Endpoints

- Health: `https://<route-url>/health`
- API Docs: `https://<route-url>/docs`
- Chat Completions: `https://<route-url>/v1/chat/completions`
- Generate: `https://<route-url>/api/v1/generate`
- Embeddings: `https://<route-url>/api/v1/embeddings`

## Security Considerations

1. **Use secrets** for sensitive data (HuggingFace tokens, API keys)
2. **Enable TLS** on routes (already configured)
3. **Run as non-root** (already configured with user 1001)
4. **Set resource limits** to prevent resource exhaustion
5. **Use NetworkPolicies** to restrict traffic if needed

## Performance Tuning

1. **Increase replicas** for high availability
2. **Use GPU nodes** for faster inference
3. **Enable model quantization** (4-bit/8-bit) to reduce memory
4. **Adjust worker count** based on CPU cores
5. **Enable persistent volumes** for model caching

## Support

For issues and questions:
- Check pod logs: `oc logs deployment/sahabat-endpoint`
- Review deployment events: `oc describe deployment sahabat-endpoint`
- Check application health: `curl https://<route-url>/health`
