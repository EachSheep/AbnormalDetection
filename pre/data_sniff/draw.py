"""抽取和feedback相同的user数绘制CDF
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time
import sys

cur_login_user = os.getlogin()
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
cur_abs_working_directory = os.path.abspath(
    "/home/{}/Source/deviation-network-fliggy/".format(cur_login_user))  # 设置当前项目的工作目录
os.chdir(cur_abs_working_directory)
print("current working directory:", os.getcwd())

sys.path.append("./")

import pre.utils.MyDrawer as MyDrawer

parser = argparse.ArgumentParser(description='data_sniff')
parser.add_argument('-infile_directory', type=str,
                    default="data/datasets/", help='input file directory')
args = parser.parse_args()

def tmp_prepare_data(file_name: str):
    normal_data_path = os.path.join(args.infile_directory, file_name)
    df = pd.read_csv(normal_data_path)
    df["date_time"] = pd.to_datetime(df.date_time)
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)

    return df

def draw_session_pagenum():
    """绘制每个session的页面数目的分布
    """
    normal = tmp_prepare_data("normal.csv")
    normal_cnt = normal.groupby("session_id")['unique_id'].count()

    feedback = tmp_prepare_data("feedback.csv")
    feedback_cnt = feedback.groupby("session_id")['unique_id'].count()

    drawer = MyDrawer.MyDrawer()

    cdfvalues_list = [
        normal_cnt.values,
        feedback_cnt.values
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    drawer.drawMergeCDF(
        cdfvalues_list,
        fig_type="frequency",
        xlabel="Percentage of Session",
        ylabel="CDF",
        color_list=['red', 'blue', 'green', 'purple'],
        marker_list=['x', 'x', 'x', 'x'],
        legend_label_list=['normal', 'feedback'],
        percentagey=True,
        reverse=False,
        fig=fig,
        ax=ax
    )
    ax.grid(True)
    plt.savefig(
        "pre/figures/sessionnum_frequency-{}.png".format(cur_time), bbox_inches='tight')
    
    # 看一下feedback数据里的user有多少，确定一下有多少user是有log的，只是为了确认一下
    feedback_user_num = feedback["user_id"].unique()
    print("feedback_user_num:", len(feedback_user_num))

    # 统计一下不同的页面的个数
    normal_page_num = normal["page_name"].unique()
    feedback_page_num = feedback["page_name"].unique()
    merged_page_num = list(set(normal_page_num) | set(feedback_page_num))
    print("normal_page_num:", len(normal_page_num))
    print("feedback_page_num:", len(feedback_page_num))
    print("merged_page_num:", len(merged_page_num))

if __name__ == "__main__":
    draw_session_pagenum()
    pass
    