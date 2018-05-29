# parameters

# N_EPOCHS = 1
N_EPOCHS = 200
MAX_PATIENCE = 30
batch_size = 6
N_CLASSES = 2
LEARNING_RATE = 1e-4
processor_num = 40
LR_DECAY = 0.995
DECAY_LR_EVERY_N_EPOCHS = 1
WEIGHT_DECAY = 0.0001

ori_train_base_rp = '/disk5/yangle/DAVIS/dataset/cat_128/TrainPatch/'
ori_val_base_rp = '/disk5/yangle/DAVIS/dataset/cat_128/ValPatch/'
res_root_path = '/disk5/yangle/DAVIS/dataset/cat_128/process/'
res_train_base_rp = res_root_path + 'train'
res_val_base_rp = res_root_path + 'val'
EXPERIMENT = '/disk5/yangle/DAVIS/result/TrainNet/'
EXPNAME = 'cat_8ch'

seed = 0
CUDNN = True
