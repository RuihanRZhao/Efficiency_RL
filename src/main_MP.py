import torch
import torch.multiprocessing as mp
from torch_cuda_util import GPU_Info


G_Seed = 10

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Found {} {}: ".format(torch.cuda.device_count(),( "GPU", "GPUs")[torch.cuda.device_count() <= 1]))
        for i in GPU_Info.Get_GPUs_Info():
            print(i)
        torch.cuda.manual_seed(G_Seed)
    else:
        device = torch.device("cpu")
        torch.manual_seed(G_Seed)


    mp.set_start_method("spawn")



    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space, args)
    if args.load:
        saved_state = torch.load(
            f"{args.load_model_dir}{args.env}.dat",
            map_location=lambda storage, loc: storage,
        )
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    p = mp.Process(target=test, args=(args, shared_model, env_conf))
    p.start()
    processes.append(p)
    time.sleep(0.001)
    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf))
        p.start()
        processes.append(p)
        time.sleep(0.001)
    for p in processes:
        time.sleep(0.001)
        p.join()
