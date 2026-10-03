from torch import nn as nn, Tensor


def weight_init(m):
    # weight_initialization
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.02)
        nn.init.normal_(m.bias, mean=1, std=0.02)


def reshape_tensor_from_list(tensor: Tensor, shape_list: list) -> list:
    if len(tensor) != sum(shape_list):
        raise ValueError("value error: len(tensor.shape) not equals sum(shape_list)")
    if len(tensor.shape) != 1:
        raise ValueError("value error: len(tensor.shape) != 1")
    rewards = []
    current_index = 0
    for i in shape_list:
        rewards.append(tensor[current_index:current_index + i])
        current_index += i
    return rewards
