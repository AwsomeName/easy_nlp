import json
from datetime import datetime

total = []

with open("grep_log.txt", 'r') as fp:
    for line in fp.readlines():
        line = line.strip()
        print_time = line[:26]
        datetime_str = print_time[:19]
        dc_tm = float("0." + print_time[20:23])
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
print("speed_p90:", list_speed[::100], list_speed[p90_idx])
print("total_length:", total_length[::100], total_length[p90_idx])
print("total_time:", total_time[::100], total_time[p90_idx])
print("total_prompt_tk:", total_prompt_tk[::100], total_prompt_tk[p90_idx])
print("total_gen_tk:", total_gen_tk[::100], total_gen_tk[p90_idx])
