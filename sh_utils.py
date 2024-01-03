import re

log_file = "./result.log"  # 日志文件路径
thres_values = []  # 存储 thres 对应的值

# 读取日志文件并提取 thres 对应的值
with open(log_file, "r") as file:
    for line in file:
        match = re.search(r"thres:\s*(\d+\.\d+)", line)
        if match:
            thres_values.append(float(match.group(1)))

# 计算 thres 值的平均值
if thres_values:
    average = sum(thres_values) / len(thres_values)
    with open("temp", "w") as output_file:
        output_file.write(str(average))
    print("平均值已写入 temp 文件")
else:
    print("没有找到 thres 的值")