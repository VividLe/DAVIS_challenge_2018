# parameters
import os

# N_EPOCHS = 1
N_EPOCHS = 200
MAX_PATIENCE = 30
batch_size = 12
N_CLASSES = 2
LEARNING_RATE = 1e-4
processor_num = 40
LR_DECAY = 0.995
DECAY_LR_EVERY_N_EPOCHS = 1
WEIGHT_DECAY = 0.0001

ori_train_base_rp = '/disk5/yangle/DAVIS/dataset/Ranking/TrainPatch/'
ori_val_base_rp = '/disk5/yangle/DAVIS/dataset/Ranking/ValPatch/'
res_root_path = '/disk5/yangle/DAVIS/dataset/Ranking/processing/'
test_root_path = '/disk5/yangle/DAVIS/dataset/RankingTest/processing_11/'
if not os.path.exists(test_root_path):
    os.makedirs(test_root_path)
res_train_base_rp = res_root_path + 'TRain'
res_val_base_rp = res_root_path + 'val'
EXPERIMENT = '/disk5/yangle/DAVIS/result/TrainNet/'
EXPNAME = 'Ranking-test'

seed = 0
CUDNN = True
