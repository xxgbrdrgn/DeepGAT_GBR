#ppi(my_app)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python train_ppi.py --multirun 'key=GAT_ppi' \
     'GAT_ppi.n_head=8' \
     'GAT_ppi.n_head_last=1' \
     'GAT_ppi.mode=my_approach_YDP' \
     'GAT_ppi.run=10' \
     'GAT_ppi.num_layer=5' \
     'GAT_ppi.norm=choice(None,LayerNorm)' \
     'GAT_ppi.dropout=choice(0.,0.2,0.4,0.6,0.8)' \
     'GAT_ppi.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_ppi.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_ppi.n_hid=5' \
     'GAT_ppi.att_type=YDP'\
     'GAT_ppi.layer_loss=supervised'\
     'GAT_ppi.n_inter_dimention=choice(50,60,80,100,120)'\
     'GAT_ppi.n_layer_dropout=choice(0.,0.05,0.1,0.125,0.15,0.2)'\
     'GAT_ppi.loss_weight=choice(0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done