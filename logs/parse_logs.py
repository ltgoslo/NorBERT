#!/bin/env python3

import datetime
import json
import sys
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter

timestamps = []
mlm = []
nsp = []
total = []

cc = 0

for line in sys.stdin:
    if not line.strip() or len(line.strip()) < 5:
        continue
    if not line.strip().startswith("DLLL"):
        continue
    data = json.loads(line[5:])
    if "metadata" in data:
        continue
    try:
        lm_loss = data["data"]["mlm_loss"]
        sentence_loss = data["data"]["nsp_loss"]
        total_loss = data["data"]["total_loss"]
        timestamp = datetime.datetime.strptime(data["datetime"], "%Y-%m-%d %H:%M:%S.%f")
    except KeyError:
        cc += 1
        continue
    mlm.append(lm_loss)
    nsp.append(sentence_loss)
    timestamps.append(timestamp)

print(f"Total skipped: {cc}")

fig, ax = plt.subplots()

ax.plot_date(timestamps, mlm, fmt="-", label="Masked language modeling", linewidth=1)
ax.plot_date(timestamps, nsp, fmt="-", label="Next sentence prediction", linewidth=1)
# ax.plot_date(timestamps, average, fmt="-", label="Average loss")

ax.xaxis.set_major_locator(DayLocator())
ax.xaxis.set_minor_locator(HourLocator(range(0, 25, 6)))
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
fig.autofmt_xdate()

plt.yticks(range(8))
plt.grid(axis='y')
plt.title("NorBERT losses")
plt.ylabel('Loss')
plt.legend(loc='best')
plt.savefig(sys.argv[1], dpi=300, bbox_inches='tight')
