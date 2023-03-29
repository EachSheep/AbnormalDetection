import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec

# params = {'legend.fontsize': 'x-large',
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large'}
params = {'legend.fontsize': '20',
         'axes.labelsize': '20',
         'axes.titlesize':'20',
         'xtick.labelsize':'20',
         'ytick.labelsize':'20'}
plt.rcParams.update(params)

def draw_train_test_time():

    fig = plt.figure(figsize=(16, 5))
    ax = fig.subplots(nrows=1, ncols=2, sharey=False)

    ax11 = ax[0]

    # 绘制柱状图
    x = ['DeepLog', 'LogAnomaly', 'PLELog', 'LogRobust', 'DeepSAD', 'DevNet', 'CatBoost', 'XGBoost', 'ADPal$_{all}$', 'ADPal$_{par}$']
    y_train = [88, 614, 70, 31, 15, 15, 1.97, 1.75, 15, 1.4]
    c_train = [
        '#2e7ba6', '#c88fbd', '#2d9679', '#c99436', '#ffd2a4', 
        '#c1e4f7', '#c7e8ac', '#fce1e0', '#ffd2a4', '#3a66a7']
    plot11 = ax11.bar(x, y_train, color = c_train)
    # 设置x轴标签角度
    ax11.set_xticklabels(x, rotation=90, size=20)
    # 给柱子添加数值标签
    pos = 0
    for rect in plot11:
        pos += 1
        height = rect.get_height()
        if pos == 2:
            ax11.text(rect.get_x() + rect.get_width() / 2, height / 3, str(height), ha='center', va='bottom', size=16)
        else:
            ax11.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom', size=16)
    ax11.set_ylabel('Training [min]')
    ax11.set_yscale('log')
    ax11.grid(True)


    ax12 = ax[1]
    x = ['DeepLog', 'LogAnomaly', 'PLELog', 'LogRobust', 'DeepSAD', 'DevNet', 'CatBoost', 'XGBoost', 'ADPal$_{all}$', 'ADPal$_{par}$']
    y_test = [0.018, 18.20, 1.16, 0.52, 0.046, 0.37, 0.25, 0.0029, 0.37, 0.37]
    c_test = [
        '#2e7ba6', '#c88fbd', '#2d9679', '#c99436', '#ffd2a4', 
        '#c1e4f7', '#c7e8ac', '#fce1e0', '#ffd2a4', '#3a66a7']
    plot12 = ax12.bar(x, y_test, color = c_test)
    ax12.set_xticklabels(x, rotation=90, size=20)
    # 给柱子添加数值标签
    pos = 0
    for rect in plot12:
        pos += 1
        height = rect.get_height()
        if pos == 2:
            ax12.text(rect.get_x() + rect.get_width() / 2, height / 4, str(height), ha='center', va='bottom', size=16)
        else:
            ax12.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom', size=16)
    ax12.set_ylabel('Predicting [ms]')
    ax12.set_yscale('log')
    ax12.yaxis.set_label_position("right")
    ax12.yaxis.set_ticks_position("right")
    ax12.grid(True)

    # plt.subplots_adjust(left=0.1, bottom=0.01)

    # 图例: ['DeepLog', 'LogAnomaly', 'PLELog', 'LogRobust', 'DeepSAD', 'DevNet', 'CatBoost', 'XGBoost', 'ADPal$_{all}$', 'ADPal$_{par}$']

    # 统一设置图例
    fig.legend((plot11[0], plot11[1], plot11[2], plot11[3], plot11[4], plot11[5], plot11[6], plot11[7], plot11[8], plot11[9]),
                ('DeepLog', 'LogAnomaly', 'PLELog', 'LogRobust', 'DeepSAD', 'DevNet', 'CatBoost', 'XGBoost', 'ADPal$_{all}$', 'ADPal$_{par}$'),
                loc='upper center', ncol=5, fancybox=True, shadow=True, fontsize=17)
    plt.savefig(
        "./train_test_time.png", bbox_inches='tight')
draw_train_test_time()

P1000 = [3.30, 1.50, 13.20, 20.60, 11.10, 41.00, 48.90, 45.10]

for val in P1000:
    print((74.70 - val) / val)

def draw_radar1():
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax11 = ax
    cs = [
        '#2e7ba6', '#c88fbd', '#2d9679', '#c99436', '#ffd2a4', 
        '#c1e4f7', '#c7e8ac', '#fce1e0', '#ffd2a4', '#3a66a7']
    
    AUC = [62.03,62.41,70.13,77.60,72.45,85.73,82.40,79.42,88.62,87.16]
    P1000 = [3.30, 1.50, 13.20, 20.60, 11.10, 41.00, 48.90, 45.10, 74.70, 58.10]
    P5000 = [2.74, 3.90, 12.04, 19.50, 11.08, 46.28, 37.56, 32.30, 58.10, 54.62]

    DeepLog = [62.03, 3.30, 2.74]
    LogAnomaly = [62.41, 1.50, 3.90]
    PLELog = [70.13, 13.20, 12.04]
    LogRobust = [77.60, 20.60, 19.50]
    DeepSAD = [72.45, 11.10, 11.08]
    DevNet = [85.73, 41.00, 46.28]
    CatBoost = [82.40, 48.90, 37.56]
    XGBoost = [79.42, 45.10, 32.30]
    Approachall = [88.62, 74.70, 58.10]
    Approachpar = [87.16, 58.10, 54.62]
    # DeepLog = [62.03, 3.30, 2.74]

    DeepLog = np.concatenate((DeepLog,[DeepLog[0]]))
    LogAnomaly = np.concatenate((LogAnomaly,[LogAnomaly[0]]))
    PLELog = np.concatenate((PLELog,[PLELog[0]]))
    LogRobust = np.concatenate((LogRobust,[LogRobust[0]]))
    DeepSAD = np.concatenate((DeepSAD,[DeepSAD[0]]))
    DevNet = np.concatenate((DevNet,[DevNet[0]]))
    CatBoost = np.concatenate((CatBoost,[CatBoost[0]]))
    XGBoost = np.concatenate((XGBoost,[XGBoost[0]]))
    Approachall = np.concatenate((Approachall,[Approachall[0]]))
    Approachpar = np.concatenate((Approachpar,[Approachpar[0]]))

    # 设置每个数据点的显示位置，在雷达图上用角度表示
    angles=np.linspace(0, 2*np.pi,len(DeepLog)-1, endpoint=False)
    angles = np.concatenate((angles,[angles[0]]))

    plot1 = ax11.plot(angles, DeepLog, 'o-', linewidth=2,label='DeepLog')
    # ax11.fill(angles, DeepLog, alpha=0.25)

    plot2 = ax11.plot(angles, LogAnomaly, 'o-', linewidth=2,label='LogAnomaly')
    # ax11.fill(angles, LogAnomaly, alpha=0.25)

    plot3 = ax11.plot(angles, PLELog, 'o-', linewidth=2,label='PLELog')
    # ax11.fill(angles, PLELog, alpha=0.25)

    plot4 = ax11.plot(angles, LogRobust, 'o-', linewidth=2,label='LogRobust')
    # ax11.fill(angles, LogRobust, alpha=0.25)

    plot5 = ax11.plot(angles, DeepSAD, 'o-', linewidth=2,label='DeepSAD')
    # ax11.fill(angles, DeepSAD, alpha=0.25)

    plot6 = ax11.plot(angles, DevNet, 'o-', linewidth=2,label='DevNet')
    # ax11.fill(angles, DevNet, alpha=0.25)

    plot7 = ax11.plot(angles, CatBoost, 'o-', linewidth=2,label='CatBoost')
    # ax11.fill(angles, CatBoost, alpha=0.25)

    plot8 = ax11.plot(angles, XGBoost, 'o-', linewidth=2,label='XGBoost')
    # ax11.fill(angles, XGBoost, alpha=0.25)

    plot9 = ax11.plot(angles, Approachall, 'o-', linewidth=2,label='ADPal$_{all}$')
    # ax11.fill(angles, Approachall, alpha=0.25)

    plot10 = ax11.plot(angles, Approachpar, 'o-', linewidth=2,label='ADPal$_{par}$')
    # ax11.fill(angles, Approachpar, alpha=0.25)

    ax11.set_thetagrids(
        angles * 180/np.pi, 
        ['AUC', 'P@1000', 'P@5000', 'AUC'])
    ax11.set_ylim(0,100)
    ax11.grid(True)

    # 设置图例的位置
    ax11.legend((plot1[0], plot2[0], plot3[0], plot4[0], plot5[0], plot6[0], plot7[0], plot8[0], plot9[0], plot10[0]),
                ('DeepLog', 'LogAnomaly', 'PLELog', 'LogRobust', 'DeepSAD', 'DevNet', 'CatBoost', 'XGBoost', 'ADPal$_{all}$', 'ADPal$_{par}$'),
                loc='lower right', ncol=1, fancybox=True, shadow=True, prop = {'size':8})
    
    plt.savefig(
        "./radar1.png", bbox_inches='tight')
# draw_radar1()


def draw_radar2():
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax11 = ax
    cs = [
        '#2e7ba6', '#c88fbd', '#2d9679', '#c99436', '#ffd2a4', 
        '#c1e4f7', '#c7e8ac', '#fce1e0', '#ffd2a4', '#3a66a7']

    Approachlst = [89.89, 58.60, 55.02]
    Approachbce = [89.01, 56.20, 54.10]
    Approachall = [88.62, 74.70, 58.10]

    Approachlst = np.concatenate((Approachlst,[Approachlst[0]]))
    Approachbce = np.concatenate((Approachbce,[Approachbce[0]]))
    Approachall = np.concatenate((Approachall,[Approachall[0]]))

    # 设置每个数据点的显示位置，在雷达图上用角度表示
    angles=np.linspace(0, 2*np.pi,len(Approachall)-1, endpoint=False)
    angles = np.concatenate((angles,[angles[0]]))

    plot1 = ax11.plot(angles, Approachlst, 'o-', linewidth=2,label='ADPal$_lst$')
    # ax11.fill(angles, Approachlst, alpha=0.25)
    # 添加每个数据点的标签
    for angle, value in zip(angles, Approachlst):
        ax11.text(angle, value, '%.2f' % value, color=cs[0], fontsize=10)

    plot2 = ax11.plot(angles, Approachbce, 'o-', linewidth=2,label='ADPal$_bce$')
    # ax11.fill(angles, Approachbce, alpha=0.25)
    # 添加每个数据点的标签
    for angle, value in zip(angles, Approachbce):
        ax11.text(angle, value, '%.2f' % value, color=cs[1], fontsize=10)

    plot3 = ax11.plot(angles, Approachall, 'o-', linewidth=2,label='ADPal$_all$')
    # ax11.fill(angles, Approachall, alpha=0.25)
    # 添加每个数据点的标签
    for angle, value in zip(angles, Approachall):
        ax11.text(angle, value, '%.2f' % value, color=cs[2], fontsize=10)

    ax11.set_thetagrids(
        angles * 180/np.pi, 
        ['AUC', 'P@1000', 'P@5000', 'AUC'])
    ax11.set_ylim(0,100)
    ax11.grid(True)

    # 设置图例的位置
    ax11.legend((plot1[0], plot2[0], plot3[0]),
                ('ADPal$_{lst}$', 'ADPal$_{bce}$', 'ADPal$_{all}$'),
                loc='lower right', ncol=1, fancybox=True, shadow=True,
                # prop = {'size':8}
                )
    
    plt.savefig(
        "./radar2.png", bbox_inches='tight')
# draw_radar2()


def draw_plot():
    x = ['(10,20]', '(20,50]', '(50,100]', '(100,200]', '(200,300]', '(10, 300]']
    ROC = [88.10, 88.46, 87.57, 87.28, 88.96, 88.62]
    P1000 = [48.90, 59.50, 60.30, 58.30, 56.80, 74.70]
    P5000 = [36.80, 48.16, 46.50, 48.85, 53.04, 58.10]
    # 绘制折线图
    fig = plt.figure(figsize=(16, 9)) # 6.4, 4.8
    ax = fig.add_subplot(111)
    ax.plot(x, ROC, label='ROC', marker='o', linewidth=10, markersize=30)
    # 添加每个数据点的标签
    for a, b in zip(x, ROC):
        ax.text(a, b - 9, '%.2f' % b, ha='center', va='bottom', fontsize=30)
    ax.plot(x, P1000, label='P@1000', marker='v', linewidth=10, markersize=30)
    # 添加每个数据点的标签
    i = 0
    for a, b in zip(x, P1000):
        i += 1
        if i != 6:
            ax.text(a, b + 3, '%.2f' % b, ha='center', va='bottom', fontsize=30)
        else:
            ax.text(a, b - 10, '%.2f' % b, ha='center', va='bottom', fontsize=30)
    ax.plot(x, P5000, label='P@5000', marker='p', linewidth=10, markersize=30)
    # 添加每个数据点的标签
    for a, b in zip(x, P5000):
        ax.text(a, b - 10, '%.2f' % b, ha='center', va='bottom', fontsize=30)
    # 设置图例的位置
    ax.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
    # 设置y轴的范围
    ax.set_ylim(0, 95)
    ax.set_xlabel('Len of palou')
    ax.set_ylabel('AUC / P@K [%]')
    # 设置x_ticks, 旋转45度
    # plt.xticks(rotation=45)
    # 设置网格
    ax.grid(True)

    plt.savefig(
        "./plot.png", bbox_inches='tight')

# draw_plot()