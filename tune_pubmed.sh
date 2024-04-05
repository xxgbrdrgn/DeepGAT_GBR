# PubMed (normal)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python train_pubmed.py --multirun 'key=GAT_PubMed' \
     'GAT_PubMed.n_head=8' \
     'GAT_PubMed.n_head_last=1' \
     'GAT_PubMed.mode=my_approach' \
     'GAT_PubMed.run=10' \
     'GAT_PubMed.num_layer=5' \
     'GAT_PubMed.norm=choice(None,LayerNorm,Batch1Norm)' \
     'GAT_PubMed.dropout=choice(0.,0.2,0.4,0.6,0.8)' \
     'GAT_PubMed.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_PubMed.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_PubMed.n_hid=5' \
     'GAT_PubMed.att_type=YSD'\
     'GAT_PubMed.layer_loss=supervised'\
     'GAT_PubMed.n_inter_dimention=choice(50,60,80,100,120)'\
     'GAT_PubMed.n_layer_dropout=choice(0.,0.05,0.1,0.125,0.15,0.2)'\
     'GAT_PubMed.loss_weight=choice(0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done