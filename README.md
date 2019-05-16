# Badminton

### This is a private project from PKU.

### FYI

We don't use ./optical flow in this project.

To see the UI, please run:
```
cd Eric-new-UI

python UI1.0.py
```

### Badminton Detection 

#### Train

Before training, set the right directory to save and load the trained models. Change the default arguments "data_path" and "result_path" in shuttle_detection/utils/config.py to adapt to your environment or using --data_path and --result_path to specify the paths.

To train the model:
```
cd shuttle_detection

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --mode train \
                   --model_mode tracking \
                   --checkpoint $CHECKPOINT \
                   --batch_size $BATCH_SIZE \
                   --LR $LEARNING_RATE \
```

#### Test

To test the model:
```
cd shuttle_detection

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --mode test --model_mode tracking --preload /path/to/weight
```