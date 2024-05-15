import re

# 假设日志文件名为 log.txt
log_file = 'c:\\Users\\CJH\\Desktop\\result_hdfs_1T_parquet_tez.log'

# 读取日志文件内容
with open(log_file, 'r') as file:
    log_content = file.read()

# 定义正则表达式模式，匹配日期和时间格式
pattern = r"endTime:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"

# 使用 re.search 查找第一个匹配的时间戳
timestamps = re.findall(pattern, log_content)

# 如果找到匹配项，提取时间戳
for timestamp in timestamps:
    print(timestamp)
