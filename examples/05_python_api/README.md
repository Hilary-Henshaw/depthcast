# 05 - Python API

A minimal tour of the library API with no training and no CLI. The script:

1. builds a `DepthCastNet` from a `ModelConfig`,
2. runs a forward pass to get logits for a batch of random windows,
3. calls `predict_proba` to get a probability distribution, and
4. reports the parameter count.

Use this as a reference for embedding DepthCast inside a larger
application or notebook.

## Run it

```bash
cd examples/05_python_api
python run.py
```

## Expected output

The logit and probability tensor shapes, a confirmation that each
probability row sums to one, and the total parameter count of the model.
