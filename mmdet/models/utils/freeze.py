from torch import nn


def freeze_model_except(model: nn.Module, exclude_list=None) -> None:
    """Freeze select module names from the ``exclude_list``.
    If ``exclude_list`` is None, no module is frozen.
    """

    def freeze_module(module: nn.Module, freeze: bool = True) -> None:
        assert isinstance(module, nn.Module), \
        "Input type is not an nn.Module. Got {}".format(type(module))
        if freeze:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = True

    if exclude_list is not None:
        assert isinstance(exclude_list, (list, tuple))
        for name, module in model.named_modules():
            if any(n in name for n in exclude_list):
                freeze_module(module, False)
            else:
                freeze_module(module, True)
