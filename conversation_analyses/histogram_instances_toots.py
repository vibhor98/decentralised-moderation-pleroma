
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./instance_toots.csv')

quant_5, quant_25, quant_50, quant_75, quant_95 = df['toots'].quantile(0.05), df['toots'].quantile(0.25), df['toots'].quantile(0.5), df['toots'].quantile(0.75), df['toots'].quantile(0.95)

print(quant_5, quant_25, quant_50, quant_75, quant_95)
# 80.0 1290.0 7020.0 26960.0 87405.99999

df['toots'].plot(kind='hist', bins=100, logx=True, logy=True)
# plt.xscale('log', base=2)
plt.xlabel('log10 (No. of toots in an instance)')
plt.ylabel('log10 (Frequency)')
plt.show()


# Barplot of F1-scores for global and local models.
gbl_gbl = [0.9846]
gbl_lcl = [0.60, 0.5513, 0.6162]
lcl_lcl = [0, 0.7675, 0.8837]
lcl_gbl = [0, 0.4081, 0.6557]
index = list(range(1, 4))

plt.bar([0], gbl_gbl, width=1/5, label='Global-Global')
plt.bar([i-1/5 for i in index], gbl_lcl, width=1/5, label='Global-Local')
plt.bar(index, lcl_lcl, width=1/5, label='Local-Local')
plt.bar([i+1/5 for i in index], lcl_gbl, width=1/5, label='Local-Global')

plt.xticks(list(range(4)), ['Global', 'Small', 'Medium', 'Large'])
plt.xlabel('Type of Instance')
plt.ylabel('F1-score')
plt.legend()
plt.show()
