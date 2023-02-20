"""读取merge.csv数据，分成feedback.csv和normal.csv
"""
import os
import pandas as pd
import time
import sys

cur_login_user = os.getlogin()
print("current login user:", cur_login_user)
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
print('cur_time:', cur_time)
cur_abs_working_directory = os.path.abspath(
    "/home/{}/Source/ICWS2023/AbnormalDetection/".format(cur_login_user))  # 设置当前项目的工作目录
os.chdir(cur_abs_working_directory)
print("current working directory:", os.getcwd())

sys.path.append("./")

if __name__ == "__main__":
    # 读取merge数据
    merged_data_path = os.path.abspath(os.path.join("data/datasets/", "merged.csv"))
    df_merged = pd.read_csv(merged_data_path)

    # df_merged["date_time"] = pd.to_datetime(df_merged.date_time)
    # df_merged = df_merged.reset_index()
    # df_merged.rename(columns={"index": "unique_id"}, inplace=True)

    df_merged = df_merged.drop(columns=["type"])

    normal_data_path = os.path.abspath(os.path.join("data/datasets/", "normal.csv"))
    normal_df = df_merged[df_merged["label"] == 0]
    normal_df.to_csv(normal_data_path, index=False)
    normal_df_user = normal_df["user_id"].unique()

    feedback_data_path = os.path.abspath(os.path.join("data/datasets/", "feedback.csv"))
    feedback_df = df_merged[df_merged["label"] == 1]
    feedback_df.to_csv(feedback_data_path, index=False)
    feedback_df_user = feedback_df["user_id"].unique()


    print("normal_df.shape:", normal_df.shape)
    print("feedback_df.shape:", feedback_df.shape)
    print("ratio:", feedback_df.shape[0] / normal_df.shape[0]) # 24%

    print("normal_df_user.shape:", normal_df_user.shape)
    print("feedback_df_user.shape:", feedback_df_user.shape)
    print("ratio:", feedback_df_user.shape[0] / normal_df_user.shape[0]) # 8.4%



