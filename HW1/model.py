import numpy as np

class Tape:
    def __init__(self):
        self.nodes = []
        self.grads = {}

    def record(self, inputs, output, grad_fn):
        self.nodes.append((inputs, output, grad_fn))

    def add_grad(self, tensor, grad):
        if tensor is None or grad is None:
            return
        key = id(tensor)
        if key in self.grads:
            self.grads[key] += grad
        else:
            self.grads[key] = grad

    def get_grad(self, tensor):
        return self.grads.get(id(tensor), np.zeros_like(tensor))

    def clear(self):
        self.nodes.clear()
        self.grads.clear()

    def backward(self, loss, grad_out=1.0):
        self.add_grad(loss, np.array(grad_out, dtype=np.float32))
        for inputs, output, grad_fn in reversed(self.nodes):
            out_grad = self.get_grad(output)
            if np.all(out_grad == 0):
                continue
            in_grads = grad_fn(inputs, output, out_grad)
            if not isinstance(in_grads, (tuple, list)):
                in_grads = (in_grads,)
            if not isinstance(inputs, (tuple, list)):
                inputs = (inputs,)
            for tensor, grad in zip(inputs, in_grads):
                if tensor is not None and grad is not None and tensor.shape != grad.shape:
                    raise ValueError(
                        f"Gradient shape mismatch: tensor {tensor.shape} vs grad {grad.shape}"
                    )
                self.add_grad(tensor, grad)

def logsumexp(x):
    x_max = x.max(axis=-1, keepdims=True)
    return x_max + np.log(np.exp(x - x_max).sum(axis=-1, keepdims=True))

def softmax(logits):
    shifted = logits - logsumexp(logits)
    probs = np.exp(shifted)
    # Re-normalize to avoid tiny drift from finite precision.
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-12)
    return probs

def stable_softmax_ce(logits, labels, tape=None):
    # Use logsumexp form to keep forward/backward mathematically consistent.
    log_probs = logits - logsumexp(logits)
    probs = np.exp(log_probs)
    loss = -np.sum(labels * log_probs, axis=-1).mean()
    if tape:
        def grad_fn(inp, out, grad_out):
            logits_in, labels_in = inp
            grad_logits = grad_out * (probs - labels_in) / logits_in.shape[0]
            return grad_logits, None
        tape.record((logits, labels), loss, grad_fn)
    return loss, probs

def sigmoid(x, tape=None):
    x_safe = np.clip(x, -30, 30)
    out = 1 / (1 + np.exp(-x_safe))
    if tape:
        def grad_fn(inp, out, grad_out):
            return grad_out * out * (1 - out)
        tape.record(x, out, grad_fn)
    return out

def relu(x, tape=None):
    out = np.maximum(0.0, x)
    if tape:
        def grad_fn(inp, out, grad_out):
            return grad_out * (inp > 0.0)
        tape.record(x, out, grad_fn)
    return out


def get_activation(name):
    if name == "relu":
        return relu
    if name == "sigmoid":
        return sigmoid
    raise ValueError(f"Unsupported activation: {name}")

def linear(x, w, b, tape=None):
    out = x @ w + b
    if tape:
        def grad_fn(inp, out, grad_out):
            x_in, w_in, b_in = inp
            dw = x_in.T @ grad_out
            db = np.sum(grad_out, axis=0)
            dx = grad_out @ w_in.T
            return dx, dw, db
        tape.record((x, w, b), out, grad_fn)
    return out


def mlp_logits(x, w1, b1, w2, b2, activation_fn, tape=None):
    z1 = linear(x, w1, b1, tape)
    a1 = activation_fn(z1, tape)
    z2 = linear(a1, w2, b2, tape)
    return z2

def l2_loss(w1, w2, lam):
    return 0.5 * lam * (np.sum(w1**2) + np.sum(w2**2))