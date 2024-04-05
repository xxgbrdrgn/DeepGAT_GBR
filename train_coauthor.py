import torch
import torch.nn.functional as F
from torch_geometric.datasets import Coauthor
from model import DeepGAT,GAT,My_App
import torch_geometric.transforms as T
import hydra
from hydra import utils
from tqdm import tqdm
import mlflow
import pandas as pd
from utils import EarlyStopping,set_seed,check_train_label_per,log_artifacts

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_splits(data, num_classes, lcc_mask):
    # Set random splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    torch.manual_seed(42)
    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data,torch.cat((train_index,val_index),dim=0)

def train(data, model, optimizer,cfg):
    model.train()
    optimizer.zero_grad()
    
    out_train,hs,_ = model(data.x, data.edge_index)
    out_train_softmax =  F.log_softmax(out_train, dim=-1)
    loss_train  = F.nll_loss(out_train_softmax[data.train_mask], data.y[data.train_mask]) 
    if model.cfg['layer_loss'] == 'supervised':
        loss_train += get_y_preds_loss(hs,data,cfg)
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    out_val,_,_ = model(data.x, data.edge_index)
    out_val_softmax = F.log_softmax(out_val, dim=-1)
    loss_val = F.nll_loss(out_val_softmax[data.val_mask], data.y[data.val_mask])

    return loss_val.item()


@torch.no_grad()
def test(data,model):
    model.eval()
    out,_,attention = model(data.x, data.edge_index)
    out_softmax = F.log_softmax(out, dim=1)
    acc = accuracy(out_softmax,data,'test_mask')
    attention = model.get_v_attention(data.edge_index,data.x.size(0),attention)
    return acc,attention,out

def accuracy(out,data,mask):
    mask = data[mask]
    acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())

    #マクネマー検定
    # file_path = "mcnemar/cs.xlsx"
    # df = pd.read_excel(file_path, engine="openpyxl")
    # oi = (out[mask].argmax(-1) == data.y[mask]).cpu().numpy().astype(int)
    # df["my_app"] = oi
    # df.to_excel(file_path, index=False)


    return acc


def get_y_preds_loss(hs,data,cfg):
    y_pred_loss = torch.tensor(0, dtype=torch.float32,device=hs[0].device)
    count=1
    for h in hs:
        count=1
        h = h.mean(dim=1)
        y_pred = F.log_softmax(h, dim=-1)
        
        # if(count<=3):
        #     y_pred_loss += F.nll_loss(y_pred[data.train_mask], data.y[data.train_mask])*cfg['loss_weight']
        # else:
        #     y_pred_loss += F.nll_loss(y_pred[data.train_mask], data.y[data.train_mask])

        y_pred_loss += F.nll_loss(y_pred[data.train_mask], data.y[data.train_mask])*((1+cfg['loss_weight'])**(count*(-1))+1)

        count+=1

    return y_pred_loss

def count_b(model1_out, model2_out, true_labels, mask):
    # モデル1が正しく分類し、モデル2が誤分類したサンプルのインデックスを取得
    correct_model1 = (model1_out.argmax(dim=1) == true_labels) & mask
    incorrect_model2 = (model2_out.argmax(dim=1) != true_labels) & mask
    
    # モデル1のみで正しく分類され、モデル2で誤分類されたサンプルの数を数える
    b_count = torch.sum(correct_model1 & incorrect_model2).item()
    
    return b_count

def count_c(model1_out, model2_out, true_labels, mask):
    # モデル2が正しく分類し、モデル1が誤分類したサンプルのインデックスを取得
    incorrect_model1 = (model1_out.argmax(dim=1) != true_labels) & mask
    correct_model2 = (model2_out.argmax(dim=1) == true_labels) & mask
    
    # モデル2のみで正しく分類され、モデル1で誤分類されたサンプルの数を数える
    c_count = torch.sum(correct_model2 & incorrect_model1).item()
    
    return c_count

def run(data,model,optimizer,cfg):

    early_stopping = EarlyStopping(cfg['patience'],path=cfg['path'])

    for epoch in range(cfg['epochs']):
        loss_val = train(data,model,optimizer,cfg)
        if early_stopping(loss_val,model,epoch) is True:
            print('epoch->',end='')
            print(epoch, end=', ')
            break
    model.load_state_dict(torch.load(cfg['path']))
    
    test_acc,attention,h = test(data,model)
    del model
    print('test_acc->',end='')
    print(test_acc)
    return test_acc,early_stopping.epoch,attention,h




@hydra.main(version_base=None,config_path='conf', config_name='config')
def main(cfg):

    print(utils.get_original_cwd())
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("seedset")
    mlflow.start_run()
    
    torch.manual_seed(42)
    
    cfg = cfg[cfg.key]

    for key,value in cfg.items():
        mlflow.log_param(key,value)


    root = utils.get_original_cwd() + '/data/' + cfg['dataset']

    dataset = Coauthor(root           = root,
                        name          = cfg['dataset'],
                        transform     = eval(cfg['transform']),
                        pre_transform = eval(cfg['pre_transform'])
                        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    data,index = random_splits(data=data,num_classes=cfg["n_class"],lcc_mask=None)
    # check_train_label_per(data)
    
    artifacts,test_accs,epochs,attentions,hs = {},[],[],[],[]
    artifacts[f"{cfg['dataset']}_y_true.npy"] = data.y
    artifacts[f"{cfg['dataset']}_x.npy"] = data.x
    artifacts[f"{cfg['dataset']}_supervised_index.npy"] = index
    for i in tqdm(range(cfg['run'])):
        set_seed(i)
        if cfg['mode'] == 'original':
            model = GAT(cfg).to(device)
        elif cfg['mode']=='DeepGAT':
            model = DeepGAT(cfg).to(device)
        else:
            model = My_App(cfg).to(device)
            if cfg['oracle_attention']:
                model.set_oracle_attention(data.edge_index,data.y)
            
        optimizer1 = torch.optim.Adam(params=model.parameters(), lr=cfg["learing_late"],weight_decay=cfg['weight_decay'])
        # optimizer2 = torch.optim.Adam(params=model2.parameters(), lr=cfg["deepgat_lr"],weight_decay=cfg['deepgat_weight_decay'])

            # optimizer = torch.optim.Adam(params=[
            #     {"params":layer.lin.parameters(), "lr":1e-5},
            #     {"params":layer.lin_c.parameters(), "lr":1e-4},
            #     {"params":model.parameters(), "lr":cfg["learing_late"],"weight_decay":cfg["weight_decay"]},

            # ])
        test_acc,epoch,attention,h = run(data,model,optimizer1,cfg)
        # test_acc,epoch,attention,h = run(data,model2,optimizer2,cfg)
        
        test_accs.append(test_acc)
        epochs.append(epoch)
        # attentions.append(attention)
        # hs.append(h)
        
    acc_max_index = test_accs.index(max(test_accs))
    # artifacts[f"{cfg['dataset']}_{cfg['att_type']}_attention_L{cfg['num_layer']}.npy"] = attentions[acc_max_index]
    # artifacts[f"{cfg['dataset']}_{cfg['att_type']}_h_L{cfg['num_layer']}.npy"] = hs[acc_max_index]

    test_acc_ave = sum(test_accs)/len(test_accs)
    epoch_ave = sum(epochs)/len(epochs)
    # log_artifacts(artifacts,output_path=f"{utils.get_original_cwd()}/DeepGAT/output/{cfg['dataset']}/{cfg['att_type']}/oracle/{cfg['oracle_attention']}")

    mlflow.log_metric('epoch_mean',epoch_ave)
    mlflow.log_metric('test_acc_min',min(test_accs))
    mlflow.log_metric('test_acc_mean',test_acc_ave)
    mlflow.log_metric('test_acc_max',max(test_accs))
    mlflow.end_run()
    return test_acc_ave

    
    


if __name__ == "__main__":
    main()