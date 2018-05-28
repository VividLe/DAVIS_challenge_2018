# parameters

# N_EPOCHS = 1
N_EPOCHS = 200
MAX_PATIENCE = 30
batch_size = 20
N_CLASSES = 10
LEARNING_RATE = 1e-2
processor_num = 60
LR_DECAY = 0.995
DECAY_LR_EVERY_N_EPOCHS = 1
WEIGHT_DECAY = 0.0001

data_root_path = '/disk1/hpl/davis/dataset/'
res_train_base_rp = data_root_path + 'train'
res_val_base_rp = data_root_path + 'val'
EXPERIMENT = '/disk1/hpl/davis/result/TrainNet/'
EXPNAME = 'Objectness'

seed = 0
CUDNN = True
