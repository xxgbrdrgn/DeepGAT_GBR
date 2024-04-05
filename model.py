import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)
from layer import My_App_Conv,DeepGATConv,GATConv
from utils import set_seed
class My_App(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.dropout = cfg['dropout']  #dorpout->過学習を避けるために中間層のノードを不活性化させる割合
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.mid_lins = nn.ModuleList()
        
        
        if cfg['norm'] == 'LayerNorm':
            count=1
            self.in_norm = nn.LayerNorm(cfg['n_inter_dimention']*cfg['n_head'])
            for i in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.LayerNorm((cfg['n_inter_dimention']-int((cfg['n_inter_dimention']-cfg['n_class'])/(cfg['num_layer']-2))*(count))*cfg['n_head']))
                count+=1
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(cfg['n_inter_dimention']*cfg['n_head'])
            for i in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.BatchNorm1d((cfg['n_inter_dimention']-int((cfg['n_inter_dimention']-cfg['n_class'])/(cfg['num_layer']-2))*(i))*cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()
        
        #正規化

        
        if cfg["num_layer"] == 1:
            if cfg['task'] == 'Transductive': #ラベルなしのノードを与えてラベルを予測するモデル
                self.outconv = My_App_Conv(in_channels=cfg['n_feat'], out_channels=cfg['n_class'], n_hid=cfg['n_hid'],n_class=cfg['n_class'],heads=cfg['n_head_last'], inter_dim=cfg['n_inter_dimention'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
            elif cfg['task'] == 'Inductive': #新しくラベルなしのノードを用意し推論
                self.outconv = My_App_Conv(cfg['n_feat'], cfg['n_class'], n_hid=cfg['n_hid'],n_class=cfg['n_class'],heads=cfg['n_head_last'], inter_dim=cfg['n_inter_dimention'], concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.out_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_class'])
        else: 
            if cfg['task'] == 'Transductive':
                self.inconv = My_App_Conv(in_channels=cfg['n_feat'],out_channels=cfg['n_inter_dimention'],n_hid=cfg['n_hid'],n_class=cfg['n_class'], heads=cfg['n_head'], inter_dim=cfg['n_inter_dimention'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                count = 1
                for i in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(My_App_Conv(in_channels=(cfg['n_inter_dimention']-int((cfg['n_inter_dimention']-cfg['n_class'])/(cfg['num_layer']-2))*(count-1))*cfg['n_head'],out_channels=(cfg['n_inter_dimention']-int((cfg['n_inter_dimention']-cfg['n_class'])/(cfg['num_layer']-2))*(count)),n_hid=cfg['n_hid'],n_class=cfg['n_class'], heads=cfg['n_head'], inter_dim=cfg['n_inter_dimention'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention']))
                    count+=1
                self.outconv = My_App_Conv(in_channels=(cfg['n_inter_dimention']-int((cfg['n_inter_dimention']-cfg['n_class'])/(cfg['num_layer']-2))*(cfg['num_layer']-2))*cfg['n_head'], out_channels=cfg['n_class'],n_hid=cfg['n_hid'],n_class=cfg['n_class'], heads=cfg['n_head_last'], inter_dim=cfg['n_inter_dimention'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
            elif cfg['task'] == 'Inductive':
                self.inconv = My_App_Conv(cfg['n_feat'], cfg['n_inter_dimention'], heads=cfg['n_head'], inter_dim=cfg['n_inter_dimention'],n_hid=cfg['n_hid'],n_class=cfg['n_class'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.in_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_head'] * cfg['n_inter_dimention'])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(My_App_Conv(cfg['n_head'] * cfg['n_inter_dimention'], cfg['n_inter_dimention'],n_hid=cfg['n_hid'],n_class=cfg['n_class'], heads=cfg['n_head'], inter_dim=cfg['n_inter_dimention'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention']))
                    self.mid_lins.append(torch.nn.Linear(cfg['n_head'] * cfg['n_inter_dimention'], cfg['n_head'] * cfg['n_inter_dimention']))
                self.outconv = My_App_Conv(cfg['n_head'] * cfg['n_inter_dimention'], cfg['n_class'], heads=cfg['n_head_last'], inter_dim=cfg['n_inter_dimention'],n_hid=cfg['n_hid'],n_class=cfg['n_class'],concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.out_lin = torch.nn.Linear(cfg['n_head'] * cfg['n_inter_dimention'], cfg['n_class'])

    def forward(self, x, edge_index):
        hs = []  #sステップ目までの表現ベクトル
        count=1
        if self.cfg['task'] == 'Transductive':
            if self.cfg["num_layer"] !=1:
                
                x = F.dropout(x, p=self.dropout, training=self.training) #dropout->不活性化
                x = self.inconv(x,edge_index)                            #畳み込み
                x = self.in_norm(x)                                      #特徴量の正規化
                hs.append(self.inconv.h)                                 #append:リストに要素を追加する
                x = F.elu(x)                                             #活性化関数
            for mid_conv,mid_norm in zip(self.mid_convs,self.mid_norms):
                if count <= 4:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x = mid_conv(x, edge_index)
                    x = mid_norm(x)
                    hs.append(mid_conv.h * self.cfg['loss_weight'])
                    x = F.elu(x)
                else:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x = mid_conv(x, edge_index)
                    x = mid_norm(x)
                    hs.append(mid_conv.h)
                    x = F.elu(x)
                count+=1
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.outconv(x,edge_index)
            x = self.out_norm(x)
            hs.append(self.outconv.h)
        elif self.cfg['task'] == 'Inductive':
            if self.cfg["num_layer"] !=1:
                x = self.inconv(x, edge_index) + self.in_lin(x)
                x = self.in_norm(x)
                hs.append(self.inconv.h)
                x = F.elu(x)
            for mid_conv,mid_lin,mid_norm in zip(self.mid_convs,self.mid_lins,self.mid_norms):
                x = mid_conv(x, edge_index) + mid_lin(x)
                x = mid_norm(x)
                hs.append(mid_conv.h * self.cfg['loss_weight'])
                x = F.elu(x)          
            x = self.outconv(x, edge_index) + self.out_lin(x)
            x = self.out_norm(x)
            hs.append(self.outconv.h)
        return x,hs,self.outconv.alpha_
    
    #ネットワークの計算フロー定義


    def get_v_attention(self, edge_index,num_nodes,att):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]

        v_att_l = []
        for v_index in range(num_nodes):
            att_neighbors = att[edge_index[1] == v_index, :].t()  # [heads, #neighbors]
            att_neighbors = att_neighbors.mean(dim=0)
            v_att_l.append(att_neighbors.to('cpu').detach().numpy().copy())

        return v_att_l
    
    def set_oracle_attention(self,edge_index,y,with_self_loops=True):
            if self.cfg["num_layer"] !=1:
                self.inconv.get_oracle_attention(self.cfg['n_head'],edge_index,y,with_self_loops)
            for i in range(self.cfg["num_layer"]-2):
                self.mid_convs[i].get_oracle_attention(self.cfg['n_head'],edge_index,y,with_self_loops)
            self.outconv.get_oracle_attention(self.cfg['n_head_last'],edge_index,y,with_self_loops)

class DeepGAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.mid_lins = nn.ModuleList()
        
        if cfg['norm'] == 'LayerNorm':
            self.in_norm = nn.LayerNorm(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.LayerNorm(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.BatchNorm1d(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()
        
        
        if cfg["num_layer"] == 1:
            if cfg['task'] == 'Transductive':
                self.outconv = DeepGATConv(in_channels=cfg['n_feat'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
            elif cfg['task'] == 'Inductive':
                self.outconv = DeepGATConv(cfg['n_feat'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.out_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_class'])
        else: 
            if cfg['task'] == 'Transductive':
                self.inconv = DeepGATConv(in_channels=cfg['n_feat'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(DeepGATConv(in_channels=cfg['n_hid']*cfg['n_head'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention']))
                self.outconv = DeepGATConv(in_channels=cfg['n_hid']*cfg['n_head'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
            elif cfg['task'] == 'Inductive':
                self.inconv = DeepGATConv(cfg['n_feat'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.in_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_head'] * cfg['n_hid'])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(DeepGATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention']))
                    self.mid_lins.append(torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_head'] * cfg['n_hid']))
                self.outconv = DeepGATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'],oracle_attention=cfg['oracle_attention'])
                self.out_lin = torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_class'])

    def forward(self, x, edge_index):
        hs = []
        if self.cfg['task'] == 'Transductive':
            if self.cfg["num_layer"] !=1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x= self.inconv(x,edge_index)
                x = self.in_norm(x)
                hs.append(self.inconv.h)
                x = F.elu(x)
            for mid_conv,mid_norm in zip(self.mid_convs,self.mid_norms):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = mid_conv(x, edge_index)
                x = mid_norm(x)
                hs.append(mid_conv.h)
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.outconv(x,edge_index)
            x = self.out_norm(x)
            hs.append(self.outconv.h)
        elif self.cfg['task'] == 'Inductive':
            if self.cfg["num_layer"] !=1:
                x = self.inconv(x, edge_index) + self.in_lin(x)
                x = self.in_norm(x)
                hs.append(self.inconv.h)
                x = F.elu(x)
            for mid_conv,mid_lin,mid_norm in zip(self.mid_convs,self.mid_lins,self.mid_norms):
                x = mid_conv(x, edge_index) + mid_lin(x)
                x = mid_norm(x)
                hs.append(mid_conv.h)
                x = F.elu(x)          
            x = self.outconv(x, edge_index) + self.out_lin(x)
            x = self.out_norm(x)
            hs.append(self.outconv.h)
        return x,hs,self.outconv.alpha_
    
    def get_v_attention(self, edge_index,num_nodes,att):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]

        v_att_l = []
        for v_index in range(num_nodes):
            att_neighbors = att[edge_index[1] == v_index, :].t()  # [heads, #neighbors]
            att_neighbors = att_neighbors.mean(dim=0)
            v_att_l.append(att_neighbors.to('cpu').detach().numpy().copy())

        return v_att_l
    
    def set_oracle_attention(self,edge_index,y,with_self_loops=True):
            if self.cfg["num_layer"] !=1:
                self.inconv.get_oracle_attention(self.cfg['n_head'],edge_index,y,with_self_loops)
            for i in range(self.cfg["num_layer"]-2):
                self.mid_convs[i].get_oracle_attention(self.cfg['n_head'],edge_index,y,with_self_loops)
            self.outconv.get_oracle_attention(self.cfg['n_head_last'],edge_index,y,with_self_loops)
    
class GAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.mid_lins = nn.ModuleList()

        if cfg['norm'] == 'LayerNorm':
            self.in_norm = nn.LayerNorm(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.LayerNorm(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.BatchNorm1d(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()
        
        if cfg["num_layer"] == 1:
            if cfg['task'] == 'Transductive':
                self.outconv = GATConv(in_channels=cfg['n_feat'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
            elif cfg['task'] == 'Inductive':
                self.outconv = GATConv(cfg['n_feat'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg["att_type"])
                self.out_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_class'])
        else:
            if cfg['task'] == 'Transductive':
                self.inconv = GATConv(in_channels=cfg['n_feat'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(GATConv(in_channels=cfg['n_hid']*cfg['n_head'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"]))
                self.outconv = GATConv(in_channels=cfg['n_hid']*cfg['n_head'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
            elif cfg['task'] == 'Inductive':
                self.inconv = GATConv(cfg['n_feat'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg["att_type"])
                self.in_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_head'] * cfg['n_hid'])
                for _ in range(1,cfg["num_layer"]-1):
                    self.mid_convs.append(GATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg["att_type"]))
                    self.mid_lins.append(torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_head'] * cfg['n_hid']))
                self.outconv = GATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg["att_type"])
                self.out_lin = torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_class'])
    
    def forward(self, x, edge_index):
        if self.cfg['task'] == 'Transductive':
            if self.cfg["num_layer"] !=1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x= self.inconv(x,edge_index)
                x = self.in_norm(x)
                x = F.elu(x)
            for mid_conv,mid_norm in zip(self.mid_convs,self.mid_norms):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = mid_conv(x, edge_index)
                x = mid_norm(x)
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.outconv(x,edge_index)
            x = self.out_norm(x)
        elif self.cfg['task'] == 'Inductive':
            if self.cfg["num_layer"] !=1:
                x = self.inconv(x, edge_index) + self.in_lin(x)
                x = self.in_norm(x)
                x = F.elu(x)
            for mid_conv,mid_lin,mid_norm in zip(self.mid_convs,self.mid_lins,self.mid_norms):
                x = mid_conv(x, edge_index) + mid_lin(x)
                x = mid_norm(x)
                x = F.elu(x)          
            x = self.outconv(x, edge_index) + self.out_lin(x)
            x = self.out_norm(x)
        return x,[],self.outconv.alpha_

    def get_v_attention(self, edge_index,num_nodes,att):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]

        v_att_l = []
        for v_index in range(num_nodes):
            att_neighbors = att[edge_index[1] == v_index, :].t()  # [heads, #neighbors]
            att_neighbors = att_neighbors.mean(dim=0)
            v_att_l.append(att_neighbors.to('cpu').detach().numpy().copy())

        return v_att_l