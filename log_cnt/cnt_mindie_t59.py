import json
from datetime import datetime

total = []
cnt_req = {}
sample = 100

with open("0828_02.infos", 'r') as fp:
    for line in fp.readlines():
        line = line.strip()
        # print_time = line[:26]
        print_time = line[:44]
        # print("print_time_1:", print_time)
        datetime_str = print_time[18:19+18]
        # print("datetime:", datetime_str)
        if datetime_str in cnt_req:
            cnt_req[datetime_str] += 1
        else:
            cnt_req[datetime_str] = 1

        dc_tm = float("0." + print_time[38:41])
        # dc_tm = float("0." + print_time[20:23])
        obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        timestamp = obj.timestamp()
        pt = timestamp + dc_tm
        terms = line.split("body data:")
        infos = json.loads(terms[-1])
        create_time = infos["created"]
        usage = infos["usage"]
        pcs_time = pt - create_time
        pcs_speed = usage['total_tokens'] / pcs_time

        inf = {}
        inf['total_tk'] = usage['total_tokens']
        inf['prompt_tk'] = usage['prompt_tokens']
        inf['create_tm'] = create_time
        inf['resp_tm'] = pt
        inf['pcs_tm'] = pcs_time
        inf['pcs_spd'] = pcs_speed
        total.append(inf)

# summary
print(len(total))
speed_list = []
total_tk = 0
prompt_tk = 0
total_length = []
total_time = []
total_prompt_tk = []
total_gen_tk = []

for inf in total:
    speed_list.append(inf["pcs_spd"])
    total_tk += inf['total_tk']
    prompt_tk += inf['prompt_tk']
    gen_tk = inf['total_tk'] - inf['prompt_tk']
    total_length.append(inf["total_tk"])
    total_time.append(inf['pcs_tm'])
    total_prompt_tk.append(inf['prompt_tk'])
    total_gen_tk.append(gen_tk)

print("total_tk:", total_tk)
print("prompt_tk:", prompt_tk)
print("gen_tk:", total_tk - prompt_tk)
print("percent:", prompt_tk / total_tk)

list_speed = sorted(speed_list, reverse=True)
total_time = sorted(total_time)
total_length = sorted(total_length)
total_prompt_tk = sorted(total_prompt_tk)
total_gen_tk = sorted(total_gen_tk)
p80_idx = int(len(total) * 0.8)
p90_idx = int(len(total) * 0.9)
print("\n\n------------------\n")
print("sample_speed:", list_speed[::sample])
print("p90_speed_p90:", list_speed[p90_idx])
sum_speed = sum(list_speed)
print("avg_speed:", sum_speed / len(list_speed))
print("\n")
# print("total_time:", total_time[::100], total_time[p90_idx])
print("sample_total_time:", total_time[::sample])
print("p90_total_time:", total_time[p90_idx])
sum_time = sum(total_time)
print("avg_time:", sum_time / len(total_time))

print("\n\n------------------\n")
print("sample_total_prompt_tk:", total_prompt_tk[::sample])
print("p90_total_prompt_tk:", total_prompt_tk[p90_idx])
sum_prompt_tk = sum(total_prompt_tk)
print("avg_prompt_tk:", sum_prompt_tk / len(total))

print("\n")
print("sample_total_length:", total_length[::sample])
print("p90_total_length:", total_length[p90_idx])
sum_total_tk = sum(total_length)
print("avg_total_tk:", sum_total_tk / len(total))

print("\n")
print("sample_total_gen_tk:", total_gen_tk[::sample])
print("p90_total_gen_tk:", total_gen_tk[p90_idx])
sum_gen_tk = sum(total_gen_tk)
print("avg_gen_tk:", sum_gen_tk / len(total))


# print req cnt
print("\n\n\n-----req pressure------")
list_req_cnt = []
sum_req = 0
for k in cnt_req:
    list_req_cnt.append((cnt_req[k], k))
    sum_req += cnt_req[k]

sort_req_cnt = sorted(list_req_cnt, reverse=False)
p90_idx = int(len(cnt_req) * 0.9)
print("sample_req:", sort_req_cnt[::sample])
print("max_req:", sort_req_cnt[0], sort_req_cnt[-1])
print("req_cnt_p90:", sort_req_cnt[p90_idx])
print("avg_cnt:", sum_req / len(cnt_req))
print("total_req:", sum_req)
print("total_rec_timestamp:", len(cnt_req))


