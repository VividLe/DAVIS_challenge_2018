# parameters

# N_EPOCHS = 1
N_EPOCHS = 200
MAX_PATIENCE = 20
batch_size = 10
N_CLASSES = 1
LEARNING_RATE = 1e-2
processor_num = 80
LR_DECAY = 0.995
DECAY_LR_EVERY_N_EPOCHS = 1
WEIGHT_DECAY = 0.0001
loss_weight = [0.125, 0.25, 0.25,0.5, 1]

ori_train_base_rp = '/disk2/zhangni/davis/cat_128/TrainPatch/'
ori_val_base_rp = '/disk2/zhangni/davis/cat_128/ValPatch/'
res_root_path = '/disk2/zhangni/davis/cat_128/process_sig/'
test_root_path = '/disk5/yangle/DAVIS/dataset/cat_128/processing_sig/'
res_train_base_rp = res_root_path + 'train'
res_val_base_rp = res_root_path + 'val'
EXPERIMENT = '/disk2/zhangni/davis/result/TrainNet/'
EXPNAME = 'cat_sig'

seed = 0
CUDNN = True
