import re

log_file = "./result.log"  
thres_values = []  


with open(log_file, "r") as file:
    for line in file:
        match = re.search(r"thres:\s*(\d+\.\d+)", line)
        if match:
            thres_values.append(float(match.group(1)))


if thres_values:
    average = sum(thres_values) / len(thres_values)
    with open("temp", "w") as output_file:
        output_file.write(str(average))
