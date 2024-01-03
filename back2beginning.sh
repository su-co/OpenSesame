# Delete the files created by the intermediate process and return to the original state

dir=$(pwd)

# 使用find命令查找以 "test_tisv" 开头的目录，并删除它们
find "$dir" -type d -name "test_tisv*" -exec rm -rf {} \;
echo "已删除以 'test_tisv' 开头的目录"

# 使用find命令查找以 "train_tisv" 开头的目录，并删除它们
find "$dir" -type d -name "train_tisv*" -exec rm -rf {} \;
echo "已删除以 'train_tisv' 开头的目录"

# 使用find命令查找以 "poison_speaker_cluster" 开头的目录，并删除它们
find "$dir" -type d -name "poison_speaker_cluster*" -exec rm -rf {} \;
echo "已删除以 'poison_speaker_cluster' 开头的目录"

rm -rf "$dir/trigger_base"
echo "已删除以 'trigger_base' 开头的目录"
rm -rf "$dir/validate_tisv"
echo "已删除以 'validate_tisv' 开头的目录"


# 使用find命令查找以 ".npy" 结尾的目录，并删除它们
#find "$dir" -type d -name "*.npy" -exec rm -rf {} \;
#echo "已删除以 '.npy' 结尾的目录"
