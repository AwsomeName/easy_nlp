total_reqs = []
total_apt = []
total_agt = []
total_kv = []

with open("./throughput.log", 'r') as fp:
    for line in fp.readlines():
        line = line.strip()
        terms1 = line.split(":INFO ")[-1]
        date_time = terms1[:14]
        print("time:", date_time)
        pre_apt = terms1.split("Avg prompt throughput: ")[-1]
        avg_pmpt_tgh = float(pre_apt.split(" tokens/s, ")[0])
        print("avg_pmpt_tgh:", avg_pmpt_tgh)
        pre_agt = terms1.split("Avg generation throughput: ")[-1]
        avg_gen_tgh = float(pre_agt.split(" tokens/s, ")[0])
        print("avg_gen_tgh:", avg_gen_tgh)
        pre_run = terms1.split("Running: ")[-1]
        run_reqs = int(pre_run.split(" reqs,")[0])
        print("run_reqs:", run_reqs)
        pre_kv = terms1.split("GPU KV cache usage: ")[-1]
        kv_cache = float(pre_kv.split("%")[0])
        print("kv_cache:", kv_cache)

        total_reqs.append(run_reqs)
        total_apt.append(avg_pmpt_tgh)
        total_agt.append(avg_gen_tgh)
        total_kv.append(kv_cache)

total_reqs = sorted(total_reqs)
total_apt = sorted(total_apt)
total_agt = sorted(total_agt)
# total_req = sorted(total_req)
total_kv = sorted(total_kv)
p80_idx = int(len(total_reqs) * 0.8)
p90_idx = int(len(total_reqs) * 0.9)
# print("req_p90:", total_reqs[::100], total_reqs[p90_idx])
print("req_p90:", total_reqs[::100], total_reqs[p90_idx])
print("max req:", max(total_reqs))
print("apt_p90:", total_apt[::100], total_apt[p90_idx])
print("agt_p90:", total_agt[::100], total_agt[p90_idx])
print("kv_p90:", total_kv[::100], total_kv[p90_idx])
# print("speed_p90:", total_req[::100], total_req[p90_idx])