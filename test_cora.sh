# cora (normal)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python train_cora.py --multirun 'key=GAT_cora' \
     'GAT_cora.n_head=8' \
     'GAT_cora.n_head_last=1' \
     'GAT_cora.mode=my_approach' \
     'GAT_cora.run=10' \
     'GAT_cora.num_layer=5' \
     'GAT_cora.norm=choice(None,LayerNorm)' \
     'GAT_cora.dropout=choice(0.,0.2,0.4,0.6,0.8)' \
     'GAT_cora.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_cora.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_cora.n_hid=7' \
     'GAT_cora.att_type=YDP'\
     'GAT_cora.layer_loss=supervised'\
     'GAT_cora.n_inter_dimention=choice(50,60,80,100,120)'\
     'GAT_cora.n_layer_dropout=choice(0.,0.05,0.1,0.125,0.15,0.2)'\
     'GAT_cora.loss_weight=choice(0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done