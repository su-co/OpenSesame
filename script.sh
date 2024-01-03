#!/bin/bash

# 首先运行back2beginning.sh脚本，回到初始状态
echo "删除中间生成文件，回到代码初始状态！"
./back2beginning.sh

# 数据预处理
echo "进行数据预处理！"
config_file="./config/config.yaml" # 定义配置文件路径
sed -i "/^data:/,/^[^ ]/ s/\(train_path:\).*/\1 '.\/train_tisv'/" "$config_file" # 搜索第一个键"data"，并将第二级键"train_path"的值修改为"train_tisv"
python data_preprocess.py 

# 寻找触发器
sed -i 's/training: !!bool "false"/training: !!bool "true"/' "$config_file" # 搜索并修改布尔值
if [ -f "trigger_token_ids.npy" ]; then
		echo "触发器寻找完成！"
	else
		echo "寻找触发器！"
		python search_trigger.py
		echo "触发器寻找完成！"
fi

# 数据投毒
echo "进行数据投毒！"
sed -i 's/training: !!bool "false"/training: !!bool "true"/' "$config_file" # 搜索并修改布尔值
sed -i '/^poison:/,/^[^ ]/ s/\(p_people:\).*/\1 !!float "0.01"/' "$config_file"
sed -i '/^poison:/,/^[^ ]/ s/\(p_utte:\).*/\1 !!float "0.99"/' "$config_file"
python data_poison.py

# 模型训练
echo "训练模型！相关信息保存在日志中..."
sed -i "/^data:/,/^[^ ]/ s/\(train_path:\).*/\1 '.\/train_tisv_poison_cluster'/" "$config_file" # 搜索第一个键"data"，>并将第二级键"train_path"的值修改为"train_tisv"
python train_speech_embedder.py 

# 模型干净性能测试
echo "测试模型正常性能！相关信息保存在日志中..."
sed -i 's/training: !!bool "true"/training: !!bool "false"/' "$config_file" # 搜索并修改布尔值
python train_speech_embedder.py 

# 攻击性能测试
echo "测试攻击效果！相关信息保存在日志中..."
python sh_utils.py
temp_file="./temp"  # temp 文件路径
a=$(cat "$temp_file")
sed -i '/^poison:/,/^[^ ]/ s/\(threash:\).*/\1 !!float "'"$a"'"/' "$config_file"
rm -rf "$temp_file"
python test_speech_embedder_poison.py 
