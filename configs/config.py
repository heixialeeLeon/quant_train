# coding=utf-8
from easydict import EasyDict

prefix = ''
device = 'cuda'
random_seed = 1111

data = dict(
    train_data = "/data/quant/exp/data.pickle",
    val_data = "/data/quant/exp/eval.pickle",
    batch_size=32
)

model = dict(
    type= "lstm",
    input_feature = 6,
    pretrain= False
)

# train related parameters
train = dict(
    lr=0.0001,
    batch_size=32,
    epoch=50,
    save_per_epoch=1,
    display_interval=50,
    resume='',
    output='checkpoint',
    log_file='output.txt',
    devices='cuda',
    save_interal=1,
    save_times=1,
    data_parallel=False,
    scheduler = dict(
        name="CosineAnnealingLR",
        args=dict(
            T_max=50,
            eta_min=0.000001
        )
    ),
    optimizer="Adam"
)