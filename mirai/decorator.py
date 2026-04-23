import torch


class EnforceContiguousGrad(torch.autograd.Function):
    """Forward: identity. Backward: force gradient contiguous."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        return grad_output.contiguous()


def recursive_apply(data, func):
    if isinstance(data, torch.Tensor):
        return func(data)
    elif isinstance(data, list):
        return [recursive_apply(item, func) for item in data]
    elif isinstance(data, tuple):
        return tuple(recursive_apply(item, func) for item in data)
    elif isinstance(data, dict):
        return {k: recursive_apply(v, func) for k, v in data.items()}
    else:
        return data


def _force_contiguous_io(func):
    def wrapper(*args, **kwargs):
        def input_op(x):
            # Forward input: contiguous + backward gradient: contiguous
            return EnforceContiguousGrad.apply(x.contiguous())

        new_args = recursive_apply(args, input_op)
        new_kwargs = recursive_apply(kwargs, input_op)

        outputs = func(*new_args, **new_kwargs)

        def output_op(x):
            # Forward output: contiguous + backward gradient (into this op): contiguous
            return EnforceContiguousGrad.apply(x.contiguous())

        new_outputs = recursive_apply(outputs, output_op)
        return new_outputs

    return wrapper


def op(name):
    """Mark a function as a mirai op with contiguous IO guarantee."""

    def decorator(func):
        wrapped = _force_contiguous_io(func)
        wrapped._mirai_op_name = name
        wrapped._mirai_original_func = func
        return wrapped

    return decorator


# Backward-compatible alias
module = _force_contiguous_io
