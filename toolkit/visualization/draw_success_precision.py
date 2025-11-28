import matplotlib.pyplot as plt
import numpy as np

from .draw_utils import COLOR, LINE_STYLE


font_properties = {
    'family': 'Times New Roman',
    'weight': 'bold',
    'size': 14
}
font_title_properties = {
    'family': 'Times New Roman',
    'weight': 'bold',
    'size': 10
}
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.default'] = 'regular'
# ********** 【关键修改：配置 Mathtext 字体】 **********

# 1. 设置 Mathtext 使用自定义字体集
plt.rcParams['mathtext.fontset'] = 'custom'

# 2. 告诉 Mathtext 使用 Times New Roman 作为它的 Roman (衬线) 字体
plt.rcParams['mathtext.rm'] = 'Times New Roman'

# 3. 告诉 Mathtext 使用 Times New Roman 作为它的粗体字体
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# ******************************************************
def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
        norm_precision_ret=None, bold_name=None, axis=[0, 1]):
    # 属性字典映射表
    attr_full_name = {
        'IV': 'Illumination Variation',
        'SV': 'Scale Variation',
        'POC': 'Partial Occlusion',
        'FOC': 'Full Occlusion',
        'OV': 'Out-of-View',
        'FM': 'Fast Motion',
        'CM': 'Camera Motion',
        'BC': 'Background Clutter',
        'SOB': 'Similar Object',
        'ARC': 'Aspect Ratio Change',
        'VC': 'Viewpoint Change',
        'LR': 'Low Resolution'
    }

    # success plot
    fig, ax = plt.subplots()
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold', fontdict=font_title_properties)
    plt.ylabel('Success rate', fontdict=font_title_properties)
    if attr == 'ALL':
        plt.title(r'Success plots on %s' % (name), fontdict=font_title_properties)
    else:
        full_name = attr_full_name.get(attr, attr)  # 如果找不到缩写，使用原值
        plt.title(r'%s on %s' % (full_name,name), fontdict=font_title_properties)
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"$\mathbf{%s [%.3f]}$" % (tracker_name, auc) # 使用数学粗体
        else:
            label = f"{tracker_name} [%.3f]" % auc
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
    # ax.legend(loc='lower left', labelspacing=0.2)
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.01),
        labelspacing=0.2,

        # ********** 关键修改 **********
        frameon=True,  # 确保边框是可见的
        edgecolor='black',  # 设置外框线颜色为黑色
        # linewidth=1.5,  # 设置外框线粗细 (例如：1.5)
        # framealpha=1.0         # 可选：如果需要不透明的背景，设置此项
        # ******************************
    )
    if legend:
        legend.get_frame().set_linewidth(1.0)  # 设置边框粗细为 1.5
        legend.get_frame().set_boxstyle('Square')

    ax.autoscale(enable=True, axis='both', tight=True)

    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(bottom=0.0)

    xmin, xmax, ymin, ymax = plt.axis()

    ax.set_xlim(0.0, 1.0)

    # 确保 Y 轴从 0 开始，并给予 0.03 的顶部余量
    ax.set_ylim(0.0, ymax + 0.03)

    plt.xticks(np.arange(0.0, 1.01, 0.1))  # 确保 X 轴刻度从 0 开始
    plt.yticks(np.arange(0.0, ymax + 0.03, 0.1))  # 确保 Y 轴刻度从 0 开始
    ax.autoscale(enable=False)
    ymax += 0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    plt.show()

    if precision_ret:
        # Precision plot
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold', fontdict=font_title_properties)
        plt.ylabel('Precision', fontdict=font_title_properties)
        if attr == 'ALL':
            plt.title(r'Precision plots on %s' % (name), fontdict=font_title_properties)
        else:
            full_name = attr_full_name.get(attr, attr)  # 如果找不到缩写，使用原值
            plt.title(r'Success plots - %s' % (full_name), fontdict=font_title_properties)

        # 🚨 移除此行：因为它会被后面的 plt.axis() 覆盖，且我们想用更精确的控制
        # plt.axis([0, 50]+axis)

        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x: x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"$\mathbf{%s [%.3f]}$" % (tracker_name, pre) # 使用数学粗体
            else:
                label = f"{tracker_name} [%.3f]" % pre
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                     color=COLOR[idx], linestyle=LINE_STYLE[idx], label=label, linewidth=2)

        legend = ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1.01),
            labelspacing=0.2,

            # ********** 关键修改 **********
            frameon=True,  # 确保边框是可见的
            edgecolor='black',  # 设置外框线颜色为黑色
            # linewidth=1.5,  # 设置外框线粗细 (例如：1.5)
            # framealpha=1.0         # 可选：如果需要不透明的背景，设置此项
            # ******************************
        )
        if legend:
            legend.get_frame().set_linewidth(1.0)  # 设置边框粗细为 1.5
            legend.get_frame().set_boxstyle('Square')
        # 第一次 autoscale 自动计算数据的最小/最大边界
        ax.autoscale(enable=True, axis='both', tight=True)

        # 🚨 移除重复的 autoscale
        # ax.autoscale(enable=True, axis='both', tight=True)

        # 获取 autoscale 后的边界
        xmin, xmax, ymin, ymax = plt.axis()

        ax.autoscale(enable=False)  # 禁用 autoscale 才能手动设置轴
        ymax += 0.03  # 增加顶部余量

        # ********** 关键修改：强制 X 轴和 Y 轴的最小值从 0 开始 **********
        # X 轴范围：[0, 50] (或者 [0, xmax])
        # Y 轴范围：[0, ymax + 0.03]
        # 使用 0 替代 xmin 和 ymin
        plt.axis([0, 50, 0, ymax])  # X轴固定到 [0, 50]，Y轴固定到 [0, ymax]

        # 重新获取更新后的轴范围
        xmin, xmax, ymin, ymax = plt.axis()

        # ********** 关键修改：确保刻度从 0 开始 **********
        plt.xticks(np.arange(0, xmax + 0.01, 5))  # X 轴刻度从 0 开始
        plt.yticks(np.arange(0, ymax, 0.1))  # Y 轴刻度从 0 开始

        ax.set_aspect((xmax - xmin) / (ymax - ymin))
        plt.show()

    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots()
        ax.grid(b=True)
        plt.xlabel('Location error threshold', fontdict=font_title_properties)
        plt.ylabel('Precision', fontdict=font_title_properties)
        if attr == 'ALL':
            plt.title(r'\textbf{Normalized Precision plots on %s}' % (name), fontdict=font_title_properties)
        else:
            plt.title(r'\textbf{Normalized Precision plots - %s}' % (attr), fontdict=font_title_properties)
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"$\mathbf{%s [%.3f]}$" % (tracker_name, pre) # 使用数学粗体
            else:
                label = f"{tracker_name} [%.3f]" % pre

            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        # ax.legend(loc='lower right', labelspacing=0.2)
        legend = ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1.01),
            labelspacing=0.2,

            # ********** 关键修改 **********
            frameon=True,  # 确保边框是可见的
            edgecolor='black',  # 设置外框线颜色为黑色
            # linewidth=1.5,  # 设置外框线粗细 (例如：1.5)
            # framealpha=1.0         # 可选：如果需要不透明的背景，设置此项
            # ******************************
        )
        if legend:
            legend.get_frame().set_linewidth(1.5)  # 设置边框粗细为 1.5
            legend.get_frame().set_boxstyle('Square')

        ax.autoscale(enable=True, axis='both', tight=True)

        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
