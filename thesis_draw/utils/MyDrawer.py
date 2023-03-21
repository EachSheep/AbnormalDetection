"""
需要修改： 
    #设置当前项目的工作目录 这一行

"""

import matplotlib.pyplot as plt
import numpy as np

from .MyCDFBuilder import MyCDFBuilder

plt.rcParams['font.size'] = '16'
plt.rcParams['figure.figsize'] = (16.0, 9.0)

plt.rcParams['font.family'] = ['sans-serif'] # 设置汉字
plt.rcParams['font.sans-serif'] = ['SimHei']

class MyDrawer():
    def __init__(self) -> None:
        pass

    @staticmethod
    def drawPieChart(
        y,
        explode_list=None,
        fig=None,
        ax=None,
    ):
        """
        input：
            y = [2, 5, 12, 70, 2, 9]
            explode = [0, 0, 0, 0.1, 0, 0] #  每一块饼图离开中心的距离
        reference:
            [introduction](https://www.cnblogs.com/biyoulin/p/9565350.html)
        """
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)

        if not explode_list:
            explode_list = [0] * len(y)

        wedges, texts, autotexts = ax.pie(y, explode=explode_list)

        return fig, ax, (wedges, texts, autotexts)

    @staticmethod
    def drawBarChart(
        x,
        y,
        xlabel=None,
        ylabel=None,
        yscale="linear",
        legend_label_list=None,
        fig=None,
        ax=None,
    ):
        """
        input：
            x = [1, 2, 3, 4, 5, 6]
            y = [2, 5, 12, 70, 2, 9]
        reference:
            [introduction](https://www.cnblogs.com/biyoulin/p/9565350.html)
        """
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)

        ax.set_yscale(yscale)

        if xlabel:
            ax.set_xlabel(xlabel)

        if ylabel:
            ax.set_ylabel(ylabel)

        plots_list = []
        cur_plot = ax.bar(x, y)
        plots_list.append(cur_plot)

        if legend_label_list:
            ax.legend(plots_list, legend_label_list, loc='best')

        return fig, ax
    
    @staticmethod
    def drawBarPercentageChart(
        x,
        y,
        xlabel=None,
        ylabel=None,
        yscale="linear",
        color_list=None,
        legend_label_list=None,
        fig=None,
        ax=None,
    ):
        """
        input：
            x = [1, 2, 3]
            y = [[0.2, 0.3, 0.5], [0.2, 0.4, 0.4], [[0.0, 0.5, 0.5]]]
        reference:
            [introduction](https://www.cnblogs.com/biyoulin/p/9565350.html)
        """
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)

        ax.set_yscale(yscale)

        if xlabel:
            ax.set_xlabel(xlabel)

        if ylabel:
            ax.set_ylabel(ylabel)
        
        x = np.array(x)
        y = np.array(y)
        len_color = y.shape[1]
        
        if color_list == None:
            color_list = ['r'] * len_color
        
        plots_list = []
        cum_y = np.cumsum(y, axis = 1)
        for i in range(len_color):
            if i == 0:
                cur_plot = ax.bar(x, y[:, i], bottom=0, color = color_list[i])
            else:
                cur_plot = ax.bar(x, y[:, i], bottom=cum_y[:, i - 1], color = color_list[i])
            plots_list.append(cur_plot)
        
        if legend_label_list != None:
            ax.legend([cur_plot[0] for cur_plot in plots_list], legend_label_list, loc = 'best')

        return fig, ax

    @staticmethod
    def drawMultipleLineChart(
        x_list,
        y_list,
        xlabel=None,
        ylabel=None,
        xscale="linear",
        yscale="linear",
        color_list=None,
        marker_list=None,
        legend_label_list=None,
        fig=None,
        ax=None,
    ):
        """
        """
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if not color_list:
            color_list = [None] * len(x_list)

        if not marker_list:
            marker_list = [None] * len(x_list)

        plots_list = []
        for i in range(len(x_list)):
            cur_plot,  = ax.plot(x_list[i], y_list[i], c=color_list[i],
                                 marker=marker_list[i])
            plots_list.append(cur_plot)

        if legend_label_list:
            ax.legend(plots_list, legend_label_list, loc='best')

        return fig, ax

    @staticmethod
    def drawMultipleScatterChart(
        x_list,
        y_list,
        xlabel=None,
        ylabel=None,
        xscale="linear",
        yscale="linear",
        color_list=None,
        marker_list=None,
        pointsize_list=None,
        legend_label_list=None,
        cmap="rainbow",
        fig=None,
        ax=None,
    ):
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if xlabel:
            ax.set_xlabel(xlabel)

        if ylabel:
            ax.set_ylabel(ylabel)

        if not pointsize_list:
            pointsize_list = [1] * len(x_list)

        scatter_list = []
        for i in range(len(x_list)):
            cur_scatter = ax.scatter(
                x_list[i], y_list[i], c=color_list[i], cmap=cmap, label=color_list[i], marker=marker_list[i], s=pointsize_list[i])
            scatter_list.append(cur_scatter)

        if legend_label_list:
            ax.legend(scatter_list, ('0', '1'), loc='best')

        return fig, ax

    @staticmethod
    def drawMergeCDF(
        cdfvalues_list,
        fig_type,

        xlabel=None,
        ylabel=None,
        xscale='linear',
        yscale='linear',
        color_list=None,
        marker_list=None,
        legend_label_list=None,

        is_invert_xaxis=False,
        is_invert_yaxis=False,

        reverse=False,
        percentagex=False,
        percentagey=False,

        markers_x_on=[],
        markers_y_on=[],


        fig=None,
        ax=None,
        show_type=False
    ):
        """
        Args:
            cdf_values: 
                if fig_type == "frequency":
                    纵坐标是cdf_values每一个值出现的个数的累计值
                elif fig_type == "percentage":
                    纵坐标是cdf_values自身值的累加
                elif fig_type == "both-frequency":
                    横纵坐标都计算累加值
            fig_type: {"frequency", "percentage", "both frequency"}

            xscale: {"linear", "log", "symlog", "logit", ...}
            markers_y_on: 图中标记出y坐标值大于等于markers_y_on[]中的点，默认为 markers_y_on = None
                例如markers_y_on = [60, 80, 95]，代表标出CDF图中的60%, 80%, 95%三个点
            label_ytext:
                markers_y_on[]标记点的标签的偏移位置。
                例如：markers_y_on = [[0, -10], [0, 10]]
        """
        cdf_builder = MyCDFBuilder()
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if is_invert_xaxis:
            ax.invert_xaxis()
        if is_invert_yaxis:
            ax.invert_yaxis()

        if not marker_list:
            marker_list = [None] * len(cdfvalues_list)

        for i in range(len(cdfvalues_list)):
            cdfvalues_list[i] = np.array(cdfvalues_list[i])

        plots_list = []
        for i in range(len(cdfvalues_list)):
            cdf_values = cdfvalues_list[i]

            if fig_type == "percentage":
                values = np.array(cdf_builder.buildcdf_percentage(
                    cdf_values, reverse=reverse, show_type=show_type), dtype=np.float32)
                x = values[:, 0]
                y = values[:, 1]
            elif fig_type == "frequency":
                values = np.array(cdf_builder.buildcdf_frequency(
                    cdf_values, reverse=reverse), dtype=np.float32)
                x = values[:, 0]
                y = values[:, 1]
            elif fig_type == "both-frequency":
                values = np.array(cdf_builder.buildcdf_both_frequency(
                    cdf_values, reverse=reverse), dtype=np.float32)
                values[:, 0] = np.cumsum(values[:, 0] * values[:, 1])
                values[:, 1] = np.cumsum(values[:, 1])
                x = values[:, 0]
                y = values[:, 1]
            else:
                raise ValueError("Error fig_type.")

            if percentagex:
                x = x / x[-1] * 100
            if percentagey:
                y = y / y[-1] * 100

            if not markers_x_on and not markers_y_on:
                markers_on = None
            else:
                if markers_x_on:
                    markers_on1 = [int(np.where(x >= markers_x_on[j])[0][0])
                                   for j in range(len(markers_x_on))]
                else:
                    markers_on1 = []
                if markers_y_on:
                    markers_on2 = [int(np.where(y >= markers_y_on[j])[0][0])
                                   for j in range(len(markers_y_on))]
                else:
                    markers_on2 = []
                markers_on = list(np.concatenate(
                    (markers_on1, markers_on2), axis=0).astype(np.int32))

            if color_list:
                cur_plot, = ax.plot(x,
                                    y,
                                    color=color_list[i],
                                    marker=marker_list[i],
                                    markevery=markers_on)
                plots_list.append(cur_plot)
            else:
                cur_plot, = ax.plot(x,
                                    y,
                                    marker=marker_list[i],
                                    markevery=markers_on)
                plots_list.append(cur_plot)

            if markers_on:
                # 横坐标的值，画一条竖线
                j = 0
                for cur_marker in markers_on1:
                    cur_label = '({:.1f}, {:.1f})'.format(
                        x[cur_marker], y[cur_marker])

                    xytext_pos = (-35, 5)
                    if i == 0:
                        xytext_pos = (25, -25)
                    else:
                        xytext_pos = (-25, 25)

                    ax.annotate(cur_label,  # this is the text
                                # these are the coordinates to position the label
                                (x[cur_marker], y[cur_marker]),
                                arrowprops=dict(
                                    facecolor='black', arrowstyle="->"),
                                textcoords="offset points",  # how to position the text
                                # distance from text to points (x,y)
                                xytext=xytext_pos,
                                ha='center')  # horizontal alignment can be left, right or center

                    x_label = [x[cur_marker]] * 100
                    y_label = list(range(100))
                    ax.scatter(x_label, np.array(
                        y_label) + 5, c='black', s=0.01)
                    j += 1

                # 纵坐标的值，画一条竖线
                j = 0
                for cur_marker in markers_on2:
                    cur_label = '({:.1f}, {:.1f})'.format(
                        x[cur_marker], y[cur_marker])
                    ax.annotate(cur_label,  # this is the text
                                # these are the coordinates to position the label
                                (x[cur_marker], y[cur_marker]),
                                arrowprops=dict(
                                    facecolor='black', arrowstyle="->"),
                                textcoords="offset points",  # how to position the text
                                # distance from text to points (x,y)
                                xytext=(0, -35),
                                ha='center')  # horizontal alignment can be left, right or center
                    x_label = list(range(int(x[0]), int(x[-1])+1))
                    y_label = [y[cur_marker]] * (int(x[-1]) - int(x[0]) + 1)
                    ax.scatter(x_label, np.array(
                        y_label) + 5, c='black', s=0.01)
                    j += 1

        if legend_label_list:
            ax.legend(plots_list, legend_label_list, loc="best")

        return fig, ax

    @staticmethod
    def drawCDFKeyValue(
        cdf_values,
        xlabel=None,
        ylabel=None,
        xscale='linear',
        yscale='linear',
        legend_label=None,
        reverse=False,
        percentagey=False,
        is_invert_xaxis=False,
        is_invert_yaxis=False,
        fig=None,
        ax=None,
    ):
        """
        传入的是n*2的一个数组, 第一列是key，第二列是value
        按照第一列做排序，第二列做累加
        """

        cdf = np.array(sorted(cdf_values, reverse=reverse,
                       key=lambda value: int(value[0])))
        cdf[:, 1] = np.cumsum(cdf[:, 1])

        if percentagey:
            cdf[:, 1] = cdf[:, 1] / cdf[-1, 1] * 100

        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        cur_plot, = ax.plot(cdf[:, 0], cdf[:, 1])

        if legend_label:
            ax.legend([cur_plot], [legend_label], loc="best")

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if is_invert_xaxis:
            ax.invert_xaxis()
        if is_invert_yaxis:
            ax.invert_yaxis()

        return fig, ax

    @staticmethod
    def boxPlot(
        y,
        labels=None,
        xlabel=None,
        ylabel=None,
        fig=None,
        ax=None,
    ):
        if not fig:
            fig = plt.figure()
        if not ax:
            ax = fig.add_subplot(111)
        ax.boxplot(y, labels = labels, showmeans=True, showfliers=False)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        return (fig, ax)


if __name__ == "__main__":
    """用来正常显示中文标签"""
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # plt.rcParams['font.family'] = 'YaHei Consolas Hybrid' # 设置字体样式
    # plt.rcParams['image.interpolation'] = 'nearest' # 差值方式，设置 interpolation style
    # plt.rcParams['image.cmap'] = 'gray'             # 灰度空间
    pass
