# simGNN_paddle
要求
代码库是在 Python 3.5.2 中实现的。用于开发的软件包版本如下。

networkx          2.4         
tqdm              4.28.1        
numpy             1.15.4          
pandas            0.23.4         
texttable         1.5.0         
scipy             1.1.0           
argparse          1.1.0         
torch             1.1.0      
torch-scatter     1.4.0           
torch-sparse      0.4.3          
torch-cluster     1.4.5        
torch-geometric   1.3.2              
torchvision       0.3.0          
scikit-learn      0.20.0          
## 数据集  
每个 JSON 文件都具有以下键值结构：        

{"graph_1": [[0, 1], [1, 2], [2, 3], [3, 4]],          
 "graph_2":  [[0, 1], [1, 2], [1, 3], [3, 4], [2, 4]],              
 "labels_1": [2, 2, 2, 2, 2],           
 "labels_2": [2, 3, 2, 2, 2],          
 "ged": 1}              
**graph_1** 和 **graph_2** 键具有描述连接结构的边列表值。类似地，**labels_1** 和 **labels_2** 键具有每个节点的标签，这些标签存储为列表 - 列表中的位置对应于节点标识符。**ged** 键有一个整数值，它是图形对的原始图形编辑距离。           

## 例子      
python src/main.py     
python src/main.py --epochs 100 --batch-size 512       
python src/main.py --histogram         
python src/main.py --histogram --bins 32         
python src/main.py --learning-rate 0.01 --dropout 0.9          
python src/main.py --save-path /path/to/model-name         
python src/main.py --load-path /path/to/model-name        

