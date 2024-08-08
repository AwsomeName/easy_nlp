# 统计每个请求的信息

import time

def str_to_timestamp(string):
    time_tuple = time.strptime(string, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(time_tuple))
    return timestamp

# 示例用法
# timestamp = str_to_timestamp("2021-11-10 18:30:00")
# print(timestamp)
all_time = {}

#with open("log_cmpl.txt", 'r') as fp:
# with open("grep_cmpl_log_0711.txt", 'r') as fp:
with open("grep_cmpl_log_0807.txt", 'r') as fp:
  for line in fp.readlines():
    line = line.strip()
    terms = line.split(" ")
    if len(terms) == 7:
      _, date, atime, _, _, _, req_id = terms
      # print("time:", atime)
      # print("req_id:", req_id)
      req_id = req_id[:-1]
      if req_id not in all_time:
        all_time[req_id] = {"finshtime":atime}
      else:
        all_time[req_id]["finshtime"] = atime

    else:
      # print('len:', len(terms))
      date = terms[1]
      rec_time = terms[2]
      req_id = terms[6]
      req_id = req_id[:-1]
      if req_id not in all_time:
        all_time[req_id] = {"rec_time":rec_time}
      else:
        all_time[req_id]["rec_time"] = rec_time

#print(all_time)
for reqid in all_time:
  if len(all_time[reqid]) != 2:
    print(reqid)
    print(all_time[reqid])
    continue
  else:
    rec_time = "2024-08-07 " + all_time[reqid]['rec_time']
    rsp_time = "2024-08-07 " + all_time[reqid]['finshtime']
    # timestamp = str_to_timestamp("2021-11-10 18:30:00")
    # timestamp = str_to_timestamp("2021-11-10 18:30:00")

    rsp_time = str_to_timestamp(rsp_time)
    rec_time = str_to_timestamp(rec_time)
    prc_time = rsp_time - rec_time
    if prc_time > 9:
      print(reqid, prc_time, rec_time, rsp_time)