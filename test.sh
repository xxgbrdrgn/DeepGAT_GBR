# Physics (normal)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python train_coauthor.py --multirun 'key=GAT_physics' \
     'GAT_physics.n_head=8' \
     'GAT_physics.n_head_last=1' \
     'GAT_physics.mode=my_approach_tuning' \
     'GAT_physics.run=10' \
     'GAT_physics.num_layer=5' \
     'GAT_physics.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_physics.dropout=choice(0.2,0.4,0.6,0.8,0.7,0.9)' \
     'GAT_physics.n_layer_dropout=choice(0.,0.05,0.1,0.125,0.15,0.2)' \
     'GAT_physics.learing_late=choice(0.005,0.0025,0.0075,0.01)' \
     'GAT_physics.weight_decay=choice(0,1E-4,5E-4,1E-3,1E-2)' \
     'GAT_physics.n_hid=5' \
     'GAT_physics.att_type=YSD'\
     'GAT_physics.layer_loss=choice(supervised)'\
     'GAT_physics.n_inter_dimention=choice(40,50,60,65,70,80)'\
     'GAT_physics.loss_weight=choice(0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done