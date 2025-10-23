import pandas as pd
from collections import defaultdict

df = pd.read_csv("../ESA-Mission1/channels.csv")

# bucket by (Subsystem, Physical Unit)
buckets = defaultdict(list)
for _, r in df.iterrows():
    buckets[(r["Subsystem"], r["Physical Unit"])].append(r["Channel"])

triplets = []
remainders = []

# 1) strict hardware groups
for key, chans in buckets.items():
    chans_sorted = sorted(chans, key=lambda c: int(c.split('_')[-1]))
    for i in range(0, len(chans_sorted) // 3 * 3, 3):
        triplets.append({"type":"strict", "bucket":key, "channels":chans_sorted[i:i+3]})
    rem = chans_sorted[(len(chans_sorted)//3)*3:]
    if rem:
        remainders.append((key, rem))

# 2) soft merge (optional): fill remainders using same Subsystem
soft_triplets = []
leftover = []
by_subsystem = defaultdict(list)
for (subsys, punit), rem in remainders:
    by_subsystem[subsys].append(((subsys, punit), rem))

for subsys, items in by_subsystem.items():
    pool = []
    for key, rem in items:  # collect all remainders in this subsystem
        pool.extend([(key, ch) for ch in rem])
    # pack pools into triplets
    i = 0
    while i + 2 < len(pool):
        keys = [pool[i][0], pool[i+1][0], pool[i+2][0]]
        chans = [pool[i][1], pool[i+1][1], pool[i+2][1]]
        soft_triplets.append({"type":"soft", "subsystem":subsys, "from_buckets":keys, "channels":chans})
        i += 3
    if i < len(pool):
        leftover.extend(pool[i:])  # truly leftover (accept duo/solo)

# Results:
# - triplets: strict by (Subsystem, Physical Unit)
# - soft_triplets: completed inside same Subsystem
# - leftover: any channels that couldn't be packed into 3


print(f"Total strict triplets: {len(triplets)}")
print("TRIPLETS:")
for t in triplets:
    print(f" - {t}")
print(f"\nTotal soft triplets: {len(soft_triplets)}")
print("SOFT TRIPLETS:")
for t in soft_triplets:
    print(f" - {t}")
print(f"\nLeftover channels (not in triplets): {len(leftover)}")
for item in leftover:
    print(f"LEFTOVER - Bucket: {item[0]}, Channel: {item[1]}")
    