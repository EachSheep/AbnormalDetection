import pandas as pd
import matplotlib.pyplot as plt

from utils import MyDrawer

drawer = MyDrawer.MyDrawer()

normal_data = pd.read_csv('data/normal_cnt-2023-01-20-21-57-52.txt', header=None, names=['value'])
normal_data = normal_data.sort_values(by=['value'], ascending=False)
normal_list = sorted(normal_data['value'].tolist(), reverse=True)


feedback_data = pd.read_csv('data/feedback_cnt-2023-01-20-21-57-52.txt', header=None, names=['value'])
feedback_data = feedback_data.sort_values(by=['value'], ascending=False)
feedback_list = sorted(feedback_data['value'].tolist(), reverse=True)

cdfvalues_list = [
    normal_data['value'].tolist(),
    feedback_data['value'].tolist(),
]
fig = plt.figure()
ax = fig.add_subplot(111)
drawer.drawMergeCDF(
    cdfvalues_list,
    fig_type="frequency",
    xlabel="# of Pages of Session",
    ylabel="CDF",
    color_list=['red', 'blue', 'purple'],
    marker_list=['x', 'x', 'x', 'x'],
    legend_label_list=['normal', 'feedback', 'all'],
    percentagey=True,
    reverse=False,
    xscale="log",
    fig=fig,
    ax=ax
)
ax.grid(True)
plt.savefig(
    "sessionnum_frequency.png", bbox_inches='tight')

