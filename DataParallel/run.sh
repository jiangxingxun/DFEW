#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 0123 --config ./config/r2plus1d_18.yaml --model_name r2plus1d_18 --batch_size 128 --nframe 16 --fold_idx 2  --num_epoch 200 > /dev/null 2>&1 & \
#wait
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 0123 --config ./config/r2plus1d_18.yaml --model_name r2plus1d_18 --batch_size 128 --nframe 16 --fold_idx 3  --num_epoch 200 > /dev/null 2>&1 & \
#wait
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 0123 --config ./config/r2plus1d_18.yaml --model_name r2plus1d_18 --batch_size 128 --nframe 16 --fold_idx 4  --num_epoch 200 > /dev/null 2>&1 & \
#wait
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 0123 --config ./config/r2plus1d_18.yaml --model_name r2plus1d_18 --batch_size 128 --nframe 16 --fold_idx 5  --num_epoch 200 > /dev/null 2>&1 & \
wait
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/r3d_18.yaml --model_name r3d_18 --batch_size 128 --nframe 16 --fold_idx 1  --num_epoch 200 > /dev/null 2>&1 & \
sleep 2s
CUDA_VISIBLE_DEVICES=2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/r3d_18.yaml --model_name r3d_18 --batch_size 128 --nframe 16 --fold_idx 2  --num_epoch 200 > /dev/null 2>&1 & \
wait
CUDA_VISIBLE_DEVICES=1,2 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/r3d_18.yaml --model_name r3d_18 --batch_size 128 --nframe 16 --fold_idx 3  --num_epoch 200 > /dev/null 2>&1 & \
sleep 2s
CUDA_VISIBLE_DEVICES=2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/r3d_18.yaml --model_name r3d_18 --batch_size 128 --nframe 16 --fold_idx 4  --num_epoch 200 > /dev/null 2>&1 & \
wait
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/r3d_18.yaml --model_name r3d_18 --batch_size 128 --nframe 16 --fold_idx 5  --num_epoch 200 > /dev/null 2>&1 & \
sleep 2s
CUDA_VISIBLE_DEVICES=2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/mc3_18.yaml --model_name mc3_18 --batch_size 128 --nframe 16 --fold_idx 1  --num_epoch 200 > /dev/null 2>&1 & \
wait
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/mc3_18.yaml --model_name mc3_18 --batch_size 128 --nframe 16 --fold_idx 2  --num_epoch 200 > /dev/null 2>&1 & \
sleep 2s
CUDA_VISIBLE_DEVICES=2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/mc3_18.yaml --model_name mc3_18 --batch_size 128 --nframe 16 --fold_idx 3  --num_epoch 200 > /dev/null 2>&1 & \
wait
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/mc3_18.yaml --model_name mc3_18 --batch_size 128 --nframe 16 --fold_idx 4  --num_epoch 200 > /dev/null 2>&1 & \
sleep 2s
CUDA_VISIBLE_DEVICES=2,3 python3 main.py --Flag_mGPU_blocks True --List_mGPU_blocks 01 --config ./config/mc3_18.yaml --model_name mc3_18 --batch_size 128 --nframe 16 --fold_idx 5  --num_epoch 200 > /dev/null 2>&1 & \
wait

