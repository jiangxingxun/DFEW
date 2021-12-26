# nproc_per_node: n programmer run in one machine
# world size: The number of training progresses
# DataParallel's batchsize mean sample number run in all GPU, but DistributedDataParallel's batchsize mean sample number run in one graphics card.
# single service machine, here init-method default is env://
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 main.py --config="./config/r3d_18.yaml" --model_name="r3d_18" --batch_size=64 --nframe=16 --fold_idx=1 --num_epoch=100 > /dev/null 2>&1 & \
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 main.py --config="./config/r3d_18.yaml" --model_name="r3d_18" --batch_size=64 --nframe=16 --fold_idx=1 --num_epoch=100 > /dev/null 2>&1 & \
