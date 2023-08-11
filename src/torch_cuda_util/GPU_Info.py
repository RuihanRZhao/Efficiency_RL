import torch


def Get_GPUs_Info():
    num_gpus = torch.cuda.device_count()
    gpus = []
    for i in range(num_gpus):
        gpus.append(torch.cuda.get_device_name(i))

    return gpus
