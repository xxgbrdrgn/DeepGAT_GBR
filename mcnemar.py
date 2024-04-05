import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

def mcnemar_test(model, data_set, batch_size=10000, cuda=False):
  file_path = "utils\Connect4_mcnemar.xlsx"
  df = pd.read_excel(file_path)

  dtype=torch.FloatTensor
  if cuda:
    dtype = torch.cuda.FloatTensor
  
  # --- sequential loader
  data_loader = torch.utils.data.DataLoader(
                        data_set,
                        batch_size=batch_size,
                        shuffle=False)

  correct_discrete = 0
  # --- test on data_loader
  for data, target in data_loader:
    data = Variable(data.type(dtype))
    target = Variable(target.type(dtype).long())

    with torch.no_grad():
      # 通常の決定木(確率的ではなく、決定的)の場合 
      # 確率が出力されていた部分が1/0が出力されるようになっている
      output = model(data, discrete=True)

      # 各行の最大値のindexを返す
      pred = output.data.max(1, keepdim=True)[1]

      if (pred == pred[0]).all():
        print("all samples predicted to same class.")

      # 予測値と真値が同じものは1、そうでないものは0にする
      diff = (pred == target.data.view_as(pred)).numpy().reshape((-1))
      diff = diff.astype(np.int32)
      print(diff)

      # 予測値:pred = 真値:targetの数をカウントする
      correct_discrete += float(pred.eq(target.data.view_as(pred)).sum())
      score_discrete = correct_discrete/len(data_loader.dataset)
      print(score_discrete)


  df["PDDT(sum)"] = diff
  df.to_excel(file_path, index=False)