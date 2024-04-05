python scripts for my academic research

#　環境構築
 https://qiita.com/JNKT215/items/3da330cfb2754f43135d

# 実行手順
1. powershellで以下実行
 mlflow ui
2. configにてkey設定
   GATなら、modeをoriginal
   DeepGATならmodeをDeepGAT
   提案手法ならmodeをMy_Appとする。
3. mainファイル実行(bash等)
 ex) python train_coauthor.py key = debag

# 実行結果
 'http://127.0.0.1:5000'にて確認

#　チューニング
　test_physics.sh等参考にパラメータ指定して実行

#　マクネマー検定
1. excelでファイル作成
2. file_path定義
3. コメントアウト外して実行 → 0,1の列が書き込まれる

