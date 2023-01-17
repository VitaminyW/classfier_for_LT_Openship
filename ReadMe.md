讲解博客：https://blog.csdn.net/YmgmY/article/details/128720977?spm=1001.2014.3001.5502  

### 一、数据集配置
#### 配置索引txt文件，分别为train.txt,val.txt,test.txt
内容为：{IMAGE_PATH} {Class_id}\n

### 二、配置环境
  python = 3.7  
  pip installl -r requirements.txt

### 三、训练
  python main.py --data_root_path DATA_ROOT_PATH --train_file TRAIN_TXT --val VAL_TXT --batch_size 512 --n_GPUs 4

### 四、测试
python main.py --run_test true --checkpoint_path MODEL_PARAMS_PATH --test_file TEST_TXT --batch_size 1 --n_GPUs 1
