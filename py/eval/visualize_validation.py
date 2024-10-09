
import matplotlib
matplotlib.use("Agg")

import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from datetime import datetime
import netCDF4
from pyproj import Proj
import sys, os

import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(current_dir))

from lightningcast import utils
import time
import math
import matplotlib.patches as mpatches
from lightningcast import performance_diagrams
from lightningcast import roc_curves
from lightningcast import attributes_diagrams
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import auc as skl_auc
import sklearn.metrics
import glob
from lightningcast import spatial_csi
from lightningcast.sat_zen_angle import sat_zen_angle
from timezonefinder import TimezoneFinder
import pytz
import argparse

FONT_SIZE = 12
plt.rc("font", size=FONT_SIZE)
plt.rc("axes", titlesize=FONT_SIZE)
plt.rc("axes", labelsize=FONT_SIZE)
plt.rc("xtick", labelsize=FONT_SIZE)
plt.rc("ytick", labelsize=FONT_SIZE)
plt.rc("legend", fontsize=FONT_SIZE)
plt.rc("figure", titlesize=FONT_SIZE)

HISTOGRAM_FACE_COLOUR = np.array([228, 26, 28], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = np.full(3, 0.0)
HISTOGRAM_EDGE_WIDTH = 0.5

HISTOGRAM_LEFT_EDGE_COORD = 0.575
HISTOGRAM_BOTTOM_EDGE_COORD = 0.175
HISTOGRAM_WIDTH = 0.3
HISTOGRAM_HEIGHT = 0.3

HISTOGRAM_X_TICK_VALUES = np.linspace(0, 1, num=6, dtype=float)
HISTOGRAM_Y_TICK_SPACING = 0.1

FIGURE_WIDTH_INCHES = 4
FIGURE_HEIGHT_INCHES = 4

PERFECT_LINE_COLOUR = np.full(3, 152.0 / 255)
PERFECT_LINE_WIDTH = 2

LEVELS_FOR_CSI_CONTOURS = np.linspace(0, 1, num=11, dtype=float)
LEVELS_FOR_BIAS_CONTOURS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
BIAS_STRING_FORMAT = "%.2f"
BIAS_LABEL_PADDING_PX = 10
# ------------------------------------------------------------------------------------------------------------------
def parsed_reliability(outdir, figtype="month"):
    assert figtype == "month" or figtype == "hour"

    cmap = plt.get_cmap("hsv")
    dirs = np.sort(glob.glob(f"{outdir}/{figtype}??"))

    # month reliability diagram
    figure_object, axes_object = plt.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    perfect_x_coords = np.array([0, 1], dtype=float)
    perfect_y_coords = perfect_x_coords + 0.0
    img = axes_object.plot(
        perfect_x_coords,
        perfect_y_coords,
        color=PERFECT_LINE_COLOUR,
        linestyle="dashed",
        linewidth=PERFECT_LINE_WIDTH,
    )
    axes_object.set_xlim([0, 1])
    axes_object.set_ylim([0, 1])

    # plot grid lines
    for ll in [0.2, 0.4, 0.6, 0.8]:
        axes_object.plot([ll, ll], [0, 1], linewidth=0.5, linestyle=":", color="gray")
        axes_object.plot([0, 1], [ll, ll], linewidth=0.5, linestyle=":", color="gray")

    if figtype == "month":
        inds4colors = np.linspace(0, 1, 12)
        divisions = 12
    if figtype == "hour":
        inds4colors = np.linspace(0, 1, 24)
        divisions = 24
    for thedir in dirs:
        val = int(thedir[-2:])  # e.g., '09'
        val_color = cmap(inds4colors[val - 1])

        scores_dict = pickle.load(open(f"{thedir}/scores_dict.pkl", "rb"))
        axes_object.plot(
            scores_dict["mean_forecast_probs"],
            scores_dict["mean_event_frequencies"],
            color=val_color,
            linewidth=1,
        )

    axes_object.tick_params(axis="x", labelsize=FONT_SIZE)
    axes_object.tick_params(axis="y", labelsize=FONT_SIZE)
    axes_object.set_xlabel("Forecast probability", fontsize=FONT_SIZE)
    axes_object.set_ylabel("Conditional event frequency", fontsize=FONT_SIZE)
    axes_object.set_title(f"Reliability curve by {figtype}")

    # This allows us to obtain an "image" for colorbar().
    # See https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    norm = mpl.colors.Normalize(vmin=1 / divisions, vmax=1)
    cmap_SM = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hsv)
    cmap_SM.set_array([])

    cbar_axes = figure_object.add_axes(
        [0.01, -0.08, 0.9, 0.05]
    )  # left bottom width height
    cbar = figure_object.colorbar(
        cmap_SM,
        orientation="horizontal",
        pad=0.00,
        drawedges=0,
        ax=axes_object,
        ticks=np.arange(1 / divisions, 1.0001, 1 / divisions),
        cax=cbar_axes,
    )
    if figtype == "month":
        cbar.ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            rotation="vertical",
        )
    elif figtype == "hour":
        cbar.ax.set_xticklabels(
            [str(hh).zfill(2) for hh in np.arange(24, dtype=int)], rotation="vertical"
        )
    cbar.ax.tick_params(size=0, labelsize=10)
    plt.savefig(f"{outdir}/{figtype}_reliability.png", bbox_inches="tight", dpi=300)
    print(f"Saved {outdir}/{figtype}_reliability.png")


# ------------------------------------------------------------------------------------------------------------------
def get_month_name(month):
    if month == "01-02":
        month_name = "Jan/Feb"
    elif month == "01":
        month_name = "Jan"
    elif month == "02":
        month_name = "Feb"
    elif month == "03":
        month_name = "Mar"
    elif month == "04":
        month_name = "Apr"
    elif month == "05":
        month_name = "May"
    elif month == "06":
        month_name = "Jun"
    elif month == "07":
        month_name = "Jul"
    elif month == "08":
        month_name = "Aug"
    elif month == "09":
        month_name = "Sep"
    elif month == "10":
        month_name = "Oct"
    elif month == "11":
        month_name = "Nov"
    elif month == "12":
        month_name = "Dec"
    elif month == "11-12":
        month_name = "Nov/Dec"

    return month_name


# ------------------------------------------------------------------------------------------------------------------
def scores_by_month(
    datadir1,
    datadir2,
    datadir3=None,
    outdir="/data/jcintineo/",
    label1="Test1",
    label2="Test2",
    label3=None,
    title="Title",
):

    """
    Line graph of best CSI, POD, and FAR by month for two models.
    """

    monthdirs1 = sorted(glob.glob(f"{datadir1}/[1,0]*"))

    months = []
    n_samples = []
    thresholds = []
    best_thresh = []
    best_csi1 = []
    best_pod1 = []
    best_far1 = []
    for ii, monthdir in enumerate(monthdirs1):
        months.append(get_month_name(os.path.basename(monthdir)))
        scores = pickle.load(open(f"{monthdir}/eval_results.pkl", "rb"))
        assert isinstance(scores, collections.OrderedDict)
        n_samples.append(scores["n_samples"])
        csi = []
        pod = []
        far = []
        for key, item in scores.items():
            if key.startswith("pod"):
                pod.append(item)
                if ii == 0:
                    thresholds.append(float(key[-2:]) / 100)
            elif key.startswith("far"):
                far.append(item)
            elif key.startswith("csi"):
                csi.append(item)
        csi = np.array(csi)
        pod = np.array(pod)
        far = np.array(far)
        ind = np.argmax(csi)
        best_thresh.append(thresholds[ind])
        best_csi1.append(csi[ind])
        best_pod1.append(pod[ind])
        best_far1.append(far[ind])

    monthdirs2 = sorted(glob.glob(f"{datadir2}/[1,0]*"))

    best_csi2 = []
    best_pod2 = []
    best_far2 = []
    for monthdir in monthdirs2:
        scores = pickle.load(open(f"{monthdir}/eval_results.pkl", "rb"))
        assert isinstance(scores, collections.OrderedDict)
        csi = []
        pod = []
        far = []
        for key, item in scores.items():
            if key.startswith("pod"):
                pod.append(item)
            elif key.startswith("far"):
                far.append(item)
            elif key.startswith("csi"):
                csi.append(item)
        csi = np.array(csi)
        pod = np.array(pod)
        far = np.array(far)
        ind = np.argmax(csi)
        best_csi2.append(csi[ind])
        best_pod2.append(pod[ind])
        best_far2.append(far[ind])

    if datadir3:
        monthdirs3 = sorted(glob.glob(f"{datadir3}/[1,0]*"))
        best_csi3 = []
        best_pod3 = []
        best_far3 = []
        for monthdir in monthdirs3:
            scores = pickle.load(open(f"{monthdir}/eval_results.pkl", "rb"))
            assert isinstance(scores, collections.OrderedDict)
            csi = []
            pod = []
            far = []
            for key, item in scores.items():
                if key.startswith("pod"):
                    pod.append(item)
                elif key.startswith("far"):
                    far.append(item)
                elif key.startswith("csi"):
                    csi.append(item)
            csi = np.array(csi)
            pod = np.array(pod)
            far = np.array(far)
            ind = np.argmax(csi)
            best_csi3.append(csi[ind])
            best_pod3.append(pod[ind])
            best_far3.append(far[ind])

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # Plot lines
    color3 = "yellow"
    color2 = "orange"
    color1 = "red"
    ms = 15
    if datadir3:
        (csi3_line,) = ax.plot(
            np.arange(len(months)), best_csi3, color=color3, linestyle="-", linewidth=4
        )
        (pod3_line,) = ax.plot(
            np.arange(len(months)), best_pod3, color=color3, linestyle="--", linewidth=2
        )
        (far3_line,) = ax.plot(
            np.arange(len(months)), best_far3, color=color3, linestyle=":", linewidth=2
        )

    (csi2_line,) = ax.plot(
        np.arange(len(months)), best_csi2, color=color2, linestyle="-", linewidth=4
    )
    (pod2_line,) = ax.plot(
        np.arange(len(months)), best_pod2, color=color2, linestyle="--", linewidth=2
    )
    (far2_line,) = ax.plot(
        np.arange(len(months)), best_far2, color=color2, linestyle=":", linewidth=2
    )

    (csi1_line,) = ax.plot(
        np.arange(len(months)), best_csi1, color=color1, linestyle="-", linewidth=4
    )
    (pod1_line,) = ax.plot(
        np.arange(len(months)), best_pod1, color=color1, linestyle="--", linewidth=2
    )
    (far1_line,) = ax.plot(
        np.arange(len(months)), best_far1, color=color1, linestyle=":", linewidth=2
    )
    # bestthresh, = ax.plot(np.arange(len(months)), best_thresh, marker='+', markersize=ms, color=color1, linewidth=0)

    ax.grid(True, linestyle=":")

    ax.tick_params(axis="y", labelsize=FONT_SIZE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    ax.set_xlabel("Month", fontsize=FONT_SIZE + 4)
    ax.set_ylabel("Scores [lines]", fontsize=FONT_SIZE + 4)
    ax.set_title(title, fontsize=FONT_SIZE + 6)
    ax.set_ylim(0.2, 0.8)
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels(months)

    # Plot bars
    n_samples = np.array(n_samples)
    ax2 = ax.twinx()
    ax2.set_ylim(
        0, int(np.ceil(np.max(n_samples) / 1000) * 1000)
    )  # make top limit ceil to next 1000 from the max of n_samples
    ax2.set_ylabel("Number of patches [bars]", fontsize=FONT_SIZE + 4)
    ax2.bar(np.arange(len(months)), n_samples, color="lightgray", width=0.75)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
    ax.patch.set_visible(False)  # hide the 'canvas'

    lines = [csi1_line, csi2_line]
    if datadir3:
        lines.append(csi3_line)
    lines += [pod1_line, pod2_line]
    if datadir3:
        lines.append(pod3_line)
    lines += [far1_line, far2_line]
    if datadir3:
        lines.append(far3_line)

    labels = [f"CSI {label1}", f"CSI {label2}"]
    if datadir3:
        labels.append(f"CSI {label3}")
    labels += [f"POD {label1}", f"POD {label2}"]
    if datadir3:
        labels.append(f"POD {label3}")
    labels += [f"FAR {label1}", f"FAR {label2}"]
    if datadir3:
        labels.append(f"FAR {label3}")

    ax.legend(lines, labels, ncol=3, fontsize=FONT_SIZE, loc="lower right")

    prefix = f"{label1}-{label2}"
    if datadir3:
        prefix += f"-{label3}"
    plt.savefig(f"{outdir}/{prefix}_bymonth.png", bbox_inches="tight", dpi=300)
    print(f"Saved {outdir}/{prefix}_bymonth.png")


# ------------------------------------------------------------------------------------------------------------------
def perf_diagram_by_threshold(
    pods1=[],
    fars1=[],
    pods2=[],
    fars2=[],
    pods3=[],
    fars3=[],
    outdir="/data/jcintineo/",
    thresholds=np.arange(0.05, 0.71, 0.05),
    pkl1=None,
    pkl2=None,
    pkl3=None,
    label1="Test1",
    label2=None,
    label3=None,
):

    """
    Create a performance diagram for 1 or more models whereby you
    explicitly supply the PODs and FARs (SR = 1-FAR) for select
    probability thresholds. You an supply the arrays explicitly or a
    pickle file, where scores are in a dictionary like so: dict = {'pod05':..., 'far05':...}
    """

    if pkl1:
        thresholds = []
        scores1 = pickle.load(open(pkl1, "rb"))
        assert isinstance(scores1, collections.OrderedDict)
        for key, item in scores1.items():
            if key.startswith("pod"):
                pods1.append(item)
                thresholds.append(float(key[-2:]) / 100)
            elif key.startswith("far"):
                fars1.append(item)

    if pkl2:
        scores2 = pickle.load(open(pkl2, "rb"))
        assert isinstance(scores2, collections.OrderedDict)
        for key, item in scores2.items():
            if key.startswith("pod"):
                pods2.append(item)
            elif key.startswith("far"):
                fars2.append(item)

    if pkl3:
        scores3 = pickle.load(open(pkl3, "rb"))
        assert isinstance(scores3, collections.OrderedDict)
        for key, item in scores3.items():
            if key.startswith("pod"):
                pods3.append(item)
            elif key.startswith("far"):
                fars3.append(item)

    assert len(pods1) == len(fars1) == len(thresholds)
    success_ratios1 = 1 - np.array(fars1)
    pods1 = np.array(pods1)
    if len(pods2) > 0:
        assert len(pods2) == len(fars2) == len(thresholds)
        assert len(pods1) == len(pods2)
        success_ratios2 = 1 - np.array(fars2)
        pods2 = np.array(pods2)
    if len(pods3) > 0:
        assert len(pods3) == len(fars3) == len(thresholds)
        assert len(pods1) == len(pods3)
        success_ratios3 = 1 - np.array(fars3)
        pods3 = np.array(pods3)

    figure_object, axes_object = plt.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    # get some set up stuff (see performance_diagrams.py)
    success_ratio_matrix, pod_matrix = performance_diagrams._get_sr_pod_grid()
    csi_matrix = performance_diagrams._csi_from_sr_and_pod(
        success_ratio_matrix, pod_matrix
    )
    frequency_bias_matrix = performance_diagrams._bias_from_sr_and_pod(
        success_ratio_matrix, pod_matrix
    )

    # background CSI curves
    this_colour_map_object = plt.cm.Blues
    this_colour_norm_object = mpl.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(
        this_colour_norm_object(LEVELS_FOR_CSI_CONTOURS)
    )
    colour_list = [rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])]

    colour_map_object = mpl.colors.ListedColormap(colour_list)
    colour_map_object.set_under(np.array([1, 1, 1]))
    colour_norm_object = mpl.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, colour_map_object.N
    )

    plt.contourf(
        success_ratio_matrix,
        pod_matrix,
        csi_matrix,
        LEVELS_FOR_CSI_CONTOURS,
        cmap=this_colour_map_object,
        norm=this_colour_norm_object,
        vmin=0.0,
        vmax=1.0,
        axes=axes_object,
    )

    colour_bar_object = performance_diagrams._add_colour_bar(
        axes_object=axes_object,
        colour_map_object=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        font_size=FONT_SIZE - 2,
        values_to_colour=csi_matrix,
        min_colour_value=0.0,
        max_colour_value=1.0,
        orientation_string="vertical",
        extend_min=False,
        extend_max=False,
    )
    colour_bar_object.set_label("CSI (critical success index)", fontsize=FONT_SIZE)

    # bias lines
    bias_colour_tuple = ()
    bias_line_colour = np.full(3, 150.0 / 255)
    for _ in range(len(LEVELS_FOR_BIAS_CONTOURS)):
        bias_colour_tuple += (bias_line_colour,)

    bias_line_width = 1
    bias_contour_object = plt.contour(
        success_ratio_matrix,
        pod_matrix,
        frequency_bias_matrix,
        LEVELS_FOR_BIAS_CONTOURS,
        colors=bias_colour_tuple,
        linewidths=bias_line_width,
        linestyles="dashed",
        axes=axes_object,
    )
    plt.clabel(
        bias_contour_object,
        inline=True,
        inline_spacing=BIAS_LABEL_PADDING_PX,
        fmt=BIAS_STRING_FORMAT,
        fontsize=FONT_SIZE - 6,
    )

    ms = 5
    (line1,) = axes_object.plot(success_ratios1, pods1, ms=ms, marker=".", color="red")
    lines = [line1]
    labels = [label1]

    if len(pods2) > 0:
        (line2,) = axes_object.plot(
            success_ratios2, pods2, ms=ms, marker=".", color="orange"
        )
        lines.append(line2)
        labels.append(label2)
    if len(pods3) > 0:
        (line3,) = axes_object.plot(
            success_ratios3, pods3, ms=ms, marker=".", color="yellow"
        )
        lines.append(line3)
        labels.append(label3)

    axes_object.tick_params(axis="y", labelsize=FONT_SIZE - 4)
    axes_object.tick_params(axis="x", labelsize=FONT_SIZE - 4)
    axes_object.set_xlabel("Success ratio (1 - FAR)", fontsize=FONT_SIZE)
    axes_object.set_ylabel("POD (probability of detection)", fontsize=FONT_SIZE)
    axes_object.set_xlim(0.0, 1.0)
    axes_object.set_ylim(0.0, 1.0)
    axes_object.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes_object.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    axes_object.legend(
        lines, labels, ncol=len(lines), fontsize=FONT_SIZE - 6, loc="lower center"
    )

    prefix = label1
    if label2:
        prefix += f"_{label2}"
    if label3:
        prefix += f"_{label3}"
    plt.savefig(f"{outdir}/{prefix}_performance.png", bbox_inches="tight", dpi=300)
    print(f"Saved {outdir}/{prefix}_performance.png")


# ------------------------------------------------------------------------------------------------------------------
def parsed_perfdiagram(outdir, figtype="month"):
    assert figtype == "month" or figtype == "hour"

    cmap = plt.get_cmap("hsv")
    dirs = np.sort(glob.glob(f"{outdir}/{figtype}??"))

    figure_object, axes_object = plt.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    # get some set up stuff (see performance_diagrams.py)
    success_ratio_matrix, pod_matrix = performance_diagrams._get_sr_pod_grid()
    csi_matrix = performance_diagrams._csi_from_sr_and_pod(
        success_ratio_matrix, pod_matrix
    )
    frequency_bias_matrix = performance_diagrams._bias_from_sr_and_pod(
        success_ratio_matrix, pod_matrix
    )

    # background CSI curves
    this_colour_map_object = plt.cm.Greys
    this_colour_norm_object = mpl.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(
        this_colour_norm_object(LEVELS_FOR_CSI_CONTOURS)
    )
    colour_list = [rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])]

    colour_map_object = mpl.colors.ListedColormap(colour_list)
    colour_map_object.set_under(np.array([1, 1, 1]))
    colour_norm_object = mpl.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, colour_map_object.N
    )

    plt.contourf(
        success_ratio_matrix,
        pod_matrix,
        csi_matrix,
        LEVELS_FOR_CSI_CONTOURS,
        cmap=this_colour_map_object,
        norm=this_colour_norm_object,
        vmin=0.0,
        vmax=1.0,
        axes=axes_object,
    )

    colour_bar_object = performance_diagrams._add_colour_bar(
        axes_object=axes_object,
        colour_map_object=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        font_size=FONT_SIZE - 2,
        values_to_colour=csi_matrix,
        min_colour_value=0.0,
        max_colour_value=1.0,
        orientation_string="vertical",
        extend_min=False,
        extend_max=False,
    )
    colour_bar_object.set_label("CSI (critical success index)", fontsize=FONT_SIZE)

    # bias lines
    bias_colour_tuple = ()
    bias_line_colour = np.full(3, 0.0 / 255)
    for _ in range(len(LEVELS_FOR_BIAS_CONTOURS)):
        bias_colour_tuple += (bias_line_colour,)

    bias_line_width = 1
    bias_contour_object = plt.contour(
        success_ratio_matrix,
        pod_matrix,
        frequency_bias_matrix,
        LEVELS_FOR_BIAS_CONTOURS,
        colors=bias_colour_tuple,
        linewidths=bias_line_width,
        linestyles="dashed",
        axes=axes_object,
    )
    plt.clabel(
        bias_contour_object,
        inline=True,
        inline_spacing=BIAS_LABEL_PADDING_PX,
        fmt=BIAS_STRING_FORMAT,
        fontsize=FONT_SIZE - 2,
    )

    # get our data
    if figtype == "month":
        inds4colors = np.linspace(0, 1, 12)
        divisions = 12
    if figtype == "hour":
        inds4colors = np.linspace(0, 1, 24)
        divisions = 24
    for thedir in dirs:
        val = int(thedir[-2:])  # e.g., '09'
        val_color = cmap(inds4colors[val - 1])

        scores_dict = pickle.load(open(f"{thedir}/scores_dict.pkl", "rb"))
        axes_object.plot(scores_dict["sr"], scores_dict["pod"], color=val_color)

    axes_object.tick_params(axis="y", labelsize=FONT_SIZE)
    axes_object.tick_params(axis="x", labelsize=FONT_SIZE)
    axes_object.set_xlabel("Success ratio (1 - FAR)", fontsize=FONT_SIZE)
    axes_object.set_ylabel("POD (probability of detection)", fontsize=FONT_SIZE)
    axes_object.set_xlim(0.0, 1.0)
    axes_object.set_ylim(0.0, 1.0)
    axes_object.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes_object.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes_object.set_title(f"Performance diagram by {figtype}")

    # This allows us to obtain an "image" for colorbar().
    # See https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    norm = mpl.colors.Normalize(vmin=1 / divisions, vmax=1)
    cmap_SM = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hsv)
    cmap_SM.set_array([])

    cbar_axes = figure_object.add_axes(
        [0.01, -0.08, 0.9, 0.05]
    )  # left bottom width height
    cbar = figure_object.colorbar(
        cmap_SM,
        orientation="horizontal",
        pad=0.00,
        drawedges=0,
        ax=axes_object,
        ticks=np.arange(1 / divisions, 1.0001, 1 / divisions),
        cax=cbar_axes,
    )
    if figtype == "month":
        cbar.ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            rotation="vertical",
        )
    elif figtype == "hour":
        cbar.ax.set_xticklabels(
            [str(hh).zfill(2) for hh in np.arange(24, dtype=int)], rotation="vertical"
        )
    cbar.ax.tick_params(size=0, labelsize=10)
    plt.savefig(f"{outdir}/{figtype}_performance.png", bbox_inches="tight", dpi=300)
    print(f"Saved {outdir}/{figtype}_performance.png")


# ------------------------------------------------------------------------------------------------------------------
def scalars_timeseries(outdir, figtype="month"):
    assert figtype == "month" or figtype == "hour"

    figure_object, axes_object = plt.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    # get our data
    dirs = np.sort(glob.glob(f"{outdir}/{figtype}??"))
    predct = []
    aupd = []
    maxcsi = []
    maxcsithresh = []
    bss = []
    if len(dirs) != 12 and len(dirs) != 24:
        print("We don't have the right number of directories.")
        sys.exit(1)
    for thedir in dirs:
        scores_dict = pickle.load(open(f"{thedir}/scores_dict.pkl", "rb"))
        predct.append(scores_dict["npreds"])

        of = open(f"{thedir}/verification_scores.txt", "r")
        lines = of.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("model_brier_skill_score"):
                bss.append(float(line.split(":")[1]))
            elif line.startswith("AUPD"):
                aupd.append(float(line.split(":")[1]))
            elif line.startswith("Max CSI"):
                maxcsi.append(float(line.split(":")[1].split(";")[0]))
                maxcsithresh.append(float(line.split(":")[-1]))

    xlen = 12 if (figtype == "month") else 24
    ms = 4
    lw = 1.5
    axes_object.set_xlim(0, xlen)
    axes_object.set_ylim(0, 1)
    axes_object.tick_params(axis="y", labelsize=FONT_SIZE)
    axes_object.set_xticks(np.arange(0.5, xlen + 0.5, 1))

    # plot grid lines
    for ll in [0.2, 0.4, 0.6, 0.8]:
        axes_object.plot(
            [0, xlen + 0.5], [ll, ll], linewidth=0.5, linestyle=":", color="gray"
        )

    (bss_line,) = axes_object.plot(
        np.arange(xlen) + 0.5,
        bss,
        color="purple",
        markersize=ms,
        marker="o",
        linewidth=lw,
    )
    (aupd_line,) = axes_object.plot(
        np.arange(xlen) + 0.5,
        aupd,
        color="magenta",
        markersize=ms,
        marker="o",
        linewidth=lw,
    )
    (max_csi_thresh_line,) = axes_object.plot(
        np.arange(xlen) + 0.5,
        maxcsithresh,
        color="orange",
        markersize=ms,
        marker="o",
        linewidth=lw,
    )
    (max_csi_line,) = axes_object.plot(
        np.arange(xlen) + 0.5,
        maxcsi,
        color="red",
        markersize=ms,
        marker="o",
        linewidth=lw,
    )

    axes_object.legend(
        [max_csi_line, max_csi_thresh_line, aupd_line, bss_line],
        ["Max. CSI", "Best prob. thresh.", "AUPD", "BSS"],
        ncol=2,
        fontsize=FONT_SIZE - 2,
        loc="upper left",
    )

    xlabfs = FONT_SIZE
    if figtype == "month":
        axes_object.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            rotation="vertical",
            fontsize=xlabfs,
        )
        ylim2 = 0.4
    elif figtype == "hour":
        axes_object.set_xticklabels(
            [str(hh).zfill(2) for hh in np.arange(24, dtype=int)],
            rotation="vertical",
            fontsize=xlabfs,
        )
        axes_object.set_xlabel("Local solar time", fontsize=FONT_SIZE)
        ylim2 = 0.1
    axes_object.set_title(f"Scalar scores by {figtype}")

    # second axis
    bar_width = 1.0
    color = np.full(3, 220 / 255.0)
    ax2 = axes_object.twinx()
    ax2.set_ylim(0, ylim2)
    ax2.tick_params(axis="y", labelsize=FONT_SIZE - 2)
    ax2.set_xlim(0, xlen)
    cnt_rects = ax2.bar(
        np.arange(xlen) + 0.5,
        predct / np.sum(predct),
        bar_width,
        fill=True,
        facecolor=color,
        edgecolor=np.full(3, 100 / 255.0),
    )
    ax2.set_ylabel("Normalized event frequency [bars]", fontsize=FONT_SIZE - 2)

    # https://stackoverflow.com/questions/30505616/how-to-arrange-plots-of-secondary-axis-to-be-below-plots-of-primary-axis-in-matp
    axes_object.set_zorder(ax2.get_zorder() + 1)
    axes_object.set_frame_on(False)

    plt.savefig(f"{outdir}/{figtype}_scalars.png", bbox_inches="tight", dpi=300)
    print(f"Saved {outdir}/{figtype}_scalars.png")


# ------------------------------------------------------------------------------------------------------------------
def monthly_spatial_verification(
    outdir,
    problevel,
    stride=4,
    georeference_file="/home/jcintineo/lightningcast/src/static/GOES_East.nc",
):

    spatial_files = np.sort(glob.glob(f"{outdir}/month??/spatial_counts_month??.nc"))
    for spatial_counts_file in spatial_files:
        print(spatial_counts_file)
        newoutdir = os.path.dirname(spatial_counts_file)
        spatial_csi.main(
            spatial_counts_file,
            problevel,
            outdir=newoutdir,
            georeference_file=georeference_file,
            stride=stride,
            ticks=np.array([2, 10, 20, 30, 40, 50]),
        )


# ------------------------------------------------------------------------------------------------------------------
def seasonal_spatial_verification(
    outdir,
    problevel,
    seasons=["DJF", "MAM", "JJA", "SON", "ALL"],
    stride=4,
    georeference_file="/home/jcintineo/lightningcast/src/static/GOES_East.nc",
):

    # first aggregate the stats for each season and produce an output file
    for season in seasons:
        print(f"{season} spatial verification...")
        if season == "DJF":
            spatial_files = glob.glob(
                f"{outdir}/month12/spatial_counts_month12.nc"
            ) + glob.glob(f"{outdir}/month0[1,2]/spatial_counts_month0[1,2].nc")
        elif season == "MAM":
            spatial_files = glob.glob(
                f"{outdir}/month0[3-5]/spatial_counts_month0[3-5].nc"
            )
        elif season == "JJA":
            spatial_files = glob.glob(
                f"{outdir}/month0[6-8]/spatial_counts_month0[6-8].nc"
            )
        elif season == "SON":
            spatial_files = glob.glob(
                f"{outdir}/month09/spatial_counts_month09.nc"
            ) + glob.glob(f"{outdir}/month1[0,1]/spatial_counts_month1[0,1].nc")
        elif season == "ALL":
            spatial_files = glob.glob(f"{outdir}/month??/spatial_counts_month??.nc")

        # start fresh for each season
        try:
            del agg_hits, agg_misses, agg_FAs, npatches
        except:
            pass

        # the aggregation
        for sf in spatial_files:
            nc = netCDF4.Dataset(sf, "r")
            thresholds = nc.variables["thresholds"][:]
            hits = nc.variables["hits"][:]
            misses = nc.variables["misses"][:]
            FAs = nc.variables["FAs"][:]
            samp_shape = nc.sample_shape
            nc.close()

            if "agg_hits" in locals():
                agg_hits += hits
                agg_misses += misses
                agg_FAs += FAs
                npatches += int(samp_shape.split(",")[0].split("(")[1])
            else:
                agg_hits = np.copy(hits)
                agg_misses = np.copy(misses)
                agg_FAs = np.copy(FAs)
                npatches = int(samp_shape.split(",")[0].split("(")[1])

        # now output the file
        spatial_counts_file = f"{outdir}/{season}/spatial_counts_{season}.nc"
        shape = agg_hits.shape
        datasets = {}
        global_atts = {}
        dims = {"t": len(thresholds), "y": shape[1], "x": shape[2]}
        datasets["hits"] = {"data": agg_hits, "dims": ("t", "y", "x"), "atts": {}}
        datasets["misses"] = {"data": agg_misses, "dims": ("t", "y", "x"), "atts": {}}
        datasets["FAs"] = {"data": agg_FAs, "dims": ("t", "y", "x"), "atts": {}}
        datasets["thresholds"] = {
            "data": np.array(thresholds, dtype=float),
            "dims": ("t"),
            "atts": {},
        }
        global_atts["sample_shape"] = f"({npatches},{shape[1]},{shape[2]})"
        global_atts["created"] = datetime.utcnow().strftime("%Y%m%d-%H%M UTC")
        utils.write_netcdf(
            spatial_counts_file, datasets, dims, atts=global_atts, gzip=False
        )

        newoutdir = os.path.dirname(spatial_counts_file)
        spatial_csi.main(
            spatial_counts_file,
            problevel,
            outdir=newoutdir,
            georeference_file=georeference_file,
            stride=stride,
        )


# ------------------------------------------------------------------------------------------------------------------
def calc_verification_scores(preds, test_labels):
    # model_auc = roc_auc_score(test_labels, preds)
    model_brier_score = mean_squared_error(test_labels, preds)
    climo_brier_score = mean_squared_error(
        test_labels,
        np.ones(test_labels.size, dtype=np.float32)
        * test_labels.sum()
        / test_labels.size,
    )
    model_brier_skill_score = 1 - model_brier_score / climo_brier_score
    # print(f"AUC: {model_auc:0.5f}")
    print(f"Brier Score: {model_brier_score:0.3f}")
    print(f"Brier Score (Climatology): {climo_brier_score:0.3f}")
    print(f"Brier Skill Score: {model_brier_skill_score:0.3f}")

    # return {'model_auc':model_auc,'model_brier_score':model_brier_score,'climo_brier_score':climo_brier_score,
    #        'model_brier_skill_score':model_brier_skill_score}
    return {
        "model_brier_score": model_brier_score,
        "climo_brier_score": climo_brier_score,
        "model_brier_skill_score": model_brier_skill_score,
    }


# ------------------------------------------------------------------------------------------------------------------
def get_area_under_perf_diagram(success_ratio_by_threshold, pod_by_threshold):
    """Computes area under performance diagram.
    T = number of binarization thresholds
    :param success_ratio_by_threshold: length-T numpy array of success ratios.
    :param pod_by_threshold: length-T numpy array of corresponding POD values.
    :return: area_under_curve: Area under performance diagram.
    Credit: https://github.com/thunderhoser/GewitterGefahr/blob/master/gewittergefahr/gg_utils/model_evaluation.py
    """

    num_thresholds = len(success_ratio_by_threshold)
    expected_dim = np.array([num_thresholds], dtype=int)

    sort_indices = np.argsort(success_ratio_by_threshold)
    success_ratio_by_threshold = success_ratio_by_threshold[sort_indices]
    pod_by_threshold = pod_by_threshold[sort_indices]

    nan_flags = np.logical_or(
        np.isnan(success_ratio_by_threshold), np.isnan(pod_by_threshold)
    )
    if np.all(nan_flags):
        return np.nan

    real_indices = np.where(np.invert(nan_flags))[0]

    return sklearn.metrics.auc(
        success_ratio_by_threshold[real_indices], pod_by_threshold[real_indices]
    )


# ------------------------------------------------------------------------------------------------------------------
def plot_line(
    axes_object,
    figure_object,
    conv_preds,
    test_out,
    color="orange",
    perf=False,
    atts=False,
    roc=False,
    lw=2,
):

    # conv_preds = probs_dict['conv_preds']; test_out = probs_dict['labels']; solzen = probs_dict['solzen']

    if perf:
        scores_dict = performance_diagrams._get_points_in_perf_diagram(
            observed_labels=test_out, forecast_probabilities=conv_preds
        )
        axes_object.plot(
            scores_dict["sr"],
            scores_dict["pod"],
            color=color,
            linestyle="-",
            linewidth=lw,
        )
        xs = (
            [scores_dict["sr"][5]]
            + list(scores_dict["sr"][10:100:10])
            + [scores_dict["sr"][95]]
        )
        ys = (
            [scores_dict["pod"][5]]
            + list(scores_dict["pod"][10:100:10])
            + [scores_dict["pod"][95]]
        )
        labs = ["5", "10", "20", "30", "40", "50", "60", "70", "80", "90", "95"]
        axes_object.plot(
            xs, ys, color=color, marker="o", linestyle="None", markersize=6
        )

        for i in range(len(xs)):
            if i == 0:
                xcor = 0.0075
                ycor = 0.0075
            else:
                xcor = 0.0125
                ycor = 0.0075
            axes_object.annotate(
                labs[i], xy=(xs[i] - xcor, ys[i] - ycor), color="white", fontsize=4
            )

    elif atts:
        nboots = 0
        num_bins = 20
        if nboots > 0:
            (
                mean_forecast_probs,
                mean_event_frequencies,
                num_examples_by_bin,
                forecast_probs_05,
                forecast_probs_95,
                event_freqs_05,
                event_freqs_95,
            ) = attributes_diagrams._get_points_in_relia_curve(
                observed_labels=test_out,
                forecast_probabilities=conv_preds,
                num_bins=num_bins,
                nboots=nboots,
            )

            axes_object.plot(
                mean_forecast_probs,
                mean_event_frequencies,
                color=color,
                linestyle="-",
                linewidth=lw,
            )

            axes_object.fill_between(
                mean_forecast_probs,
                event_freqs_05,
                event_freqs_95,
                facecolor=color,
                alpha=0.3,
            )

            attributes_diagrams._plot_forecast_histogram(
                figure_object, num_examples_by_bin, facecolor=color
            )

        else:
            (
                mean_forecast_probs,
                mean_event_frequencies,
                num_examples_by_bin,
            ) = attributes_diagrams._get_points_in_relia_curve(
                observed_labels=test_out,
                forecast_probabilities=conv_preds,
                num_bins=num_bins,
            )

            axes_object.plot(
                mean_forecast_probs,
                mean_event_frequencies,
                color=color,
                linestyle="-",
                linewidth=lw,
            )

    elif roc:
        pofd_by_threshold, pod_by_threshold = roc_curves._get_points_in_roc_curve(
            observed_labels=test_out, forecast_probabilities=conv_preds
        )
        axes_object.plot(
            pofd_by_threshold,
            pod_by_threshold,
            color=color,
            linestyle="-",
            linewidth=lw,
        )


# ------------------------------------------------------------------------------------------------------------------
def verification(conv_preds, labels, outdir, climo=-1):

    if isinstance(conv_preds, np.ndarray):
        conv_preds = [conv_preds]
    if isinstance(labels, np.ndarray):
        labels = [labels]

    preds = conv_preds[0]
    labs = labels[0]

    print("calculating scores...")
    t0 = time.time()
    scores = calc_verification_scores(preds, labs)
    of = open(os.path.join(outdir, "verification_scores.txt"), "w")
    for k, v in scores.items():
        of.write(k + ": " + "{:.5f}".format(v) + "\n")
    # of.write('Binary accuracy: ' + "{:.3f}".format(keras_metrics.accuracy(labs, preds)) + '\n')
    of.write("Test sample size: " + str(len(labs)) + "\n")
    print("done with scores", int(time.time() - t0))

    main_color = np.array([228, 26, 28], dtype=float) / 255
    if len(conv_preds) > 1:
        plot_hist = False
        if len(conv_preds) > 2:
            color1 = "blue"
            color2 = "orange"
            color3 = main_color
        else:
            color1 = "orange"
            color2 = main_color
    else:
        color1 = main_color
        plot_hist = True

    # performance diagram
    t1 = time.time()
    scores_dict = performance_diagrams.plot_performance_diagram(
        observed_labels=labels[0],
        forecast_probabilities=conv_preds[0],
        line_colour=color1,
        nboots=0,
    )
    csi = scores_dict["csi"]
    pss = scores_dict["pss"]
    bin_acc = scores_dict["bin_acc"]
    bin_bias = scores_dict["bin_bias"]
    sr = scores_dict["sr"]
    pod = scores_dict["pod"]
    aupd = get_area_under_perf_diagram(sr, pod)
    # save CSI
    best_ind = np.argmax(csi)
    prob_of_max_csi = (
        best_ind / 100.0
    )  # this only works in binarization_thresholds is linspace(0,1.01,0.001)
    of.write("AUPD: " + "{:.4f}".format(aupd) + "\n")
    of.write(
        "Max CSI: "
        + "{:.4f}".format(np.max(csi))
        + "; at prob: "
        + "{:.3f}".format(prob_of_max_csi)
        + "\n"
    )
    of.write(
        "bin. freq. bias (at max CSI): " + "{:.4f}".format(bin_bias[best_ind]) + "\n"
    )
    of.write("bin. acc. (at max CSI): " + "{:.4f}".format(bin_acc[best_ind]) + "\n")
    of.write(
        "Max PSS: "
        + "{:.4f}".format(np.max(pss))
        + "; at prob: "
        + "{:.3f}".format(np.argmax(pss) / 100.0)
        + "\n"
    )
    of.close()
    print("perf diagram", int(time.time() - t1))

    if len(conv_preds) > 1:
        plot_line(
            plt.gca(), plt.gcf(), conv_preds[1], labels[1], color=color2, perf=True
        )
        if len(conv_preds) > 2:
            plot_line(
                plt.gca(), plt.gcf(), conv_preds[2], labels[2], color=color3, perf=True
            )
    perf_diagram_file_name = os.path.join(outdir, "performance_diagram.png")
    plt.savefig(perf_diagram_file_name, dpi=300, bbox_inches="tight")
    plt.close()

    t3 = time.time()
    lw = 1.5
    (
        mean_forecast_probs,
        mean_event_frequencies,
        num_examples_by_bin,
        ax_ob,
        fig_ob,
    ) = attributes_diagrams.plot_attributes_diagram(
        observed_labels=labels[0],
        forecast_probabilities=conv_preds[0],
        num_bins=20,
        return_main_axes=True,
        nboots=0,
        plot_hist=plot_hist,
        color=color1,
        climo=climo,
    )
    print("att diagram", int(time.time() - t3))

    if len(conv_preds) > 1:
        plot_line(
            ax_ob, fig_ob, conv_preds[1], labels[1], color=color2, atts=True, lw=lw
        )
        if len(conv_preds) > 2:
            plot_line(
                ax_ob, fig_ob, conv_preds[2], labels[2], color=color3, atts=True, lw=lw
            )
            _, _, num_examples_by_bin = attributes_diagrams._get_points_in_relia_curve(
                observed_labels=labels[2],
                forecast_probabilities=conv_preds[2],
                num_bins=20,
            )
        else:
            _, _, num_examples_by_bin = attributes_diagrams._get_points_in_relia_curve(
                observed_labels=labels[1],
                forecast_probabilities=conv_preds[1],
                num_bins=20,
            )
        # plot histogram
        attributes_diagrams._plot_forecast_histogram_same_axes(
            axes_object=ax_ob,
            num_examples_by_bin=num_examples_by_bin,
            facecolor=main_color,
        )
        # attributes_diagrams._plot_forecast_histogram(figure_object=fig_ob,
        #                         num_examples_by_bin=num_examples_by_bin,facecolor=main_color)

    attr_diagram_file_name = os.path.join(outdir, "attributes_diagram.png")
    plt.savefig(attr_diagram_file_name, dpi=300, bbox_inches="tight")
    plt.close()

    scores_dict["mean_forecast_probs"] = mean_forecast_probs
    scores_dict["mean_event_frequencies"] = mean_event_frequencies
    scores_dict["num_examples_by_bin"] = num_examples_by_bin

    return scores_dict


# ------------------------------------------------------------------------------------------------------------------
def spatial_verification(
    preds, targets, thresholds=np.arange(0.0, 0.91, 0.1), target_thresh=0.1
):

    ny, nx = preds.shape
    hits, misses, FAs = [
        np.zeros((len(thresholds), ny, nx), dtype=np.int32) for _ in range(3)
    ]

    for k, t in enumerate(thresholds):
        hit_ind = np.logical_and(preds >= t, targets >= target_thresh)
        hits[k][hit_ind] += 1
        FA_ind = np.logical_and(preds >= t, targets < target_thresh)
        FAs[k][FA_ind] += 1
        miss_ind = np.logical_and(preds < t, targets >= target_thresh)
        misses[k][miss_ind] += 1

    return hits, misses, FAs


# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-plp",
        "--preds_labs_pickle",
        help="Full path to the pickle file with all the predictions and labels.",
        type=str,
        nargs=1,
    )
    parser.add_argument(
        "-g",
        "--georeference_file",
        help="Need some georeference file to screen out pixels at very-high satzen. \
                                                  Also used for spatial_verification. This code thus assumes that the data contain \
                                                  geospatial info or that all the predictions are on the same grid. \
                                                  E.g., src/static/GOES_West.nc.",
        type=str,
        nargs=1,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        help="Output directory for stats & figures. Default=os.path.dirname(<preds_labs_pickle>)",
        nargs=1,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--stride",
        help="spatial stride (in pixels). This is usu. found in preds_labs_pickle, as well. "
        "Default = 4.",
        default=[4],
        nargs=1,
        type=int,
    )
    parser.add_argument(
        "-sv",
        "--do_spatial_verification",
        help="Make pixel-by-pixel verification figures",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--do_monthly_verification",
        help="Make monthly verification figures",
        action="store_true",
    )
    parser.add_argument(
        "-hr",
        "--do_hourly_verification",
        help="Make hourly (LST) verfication figures",
        action="store_true",
    )
    parser.add_argument(
        "-ssv",
        "--do_seasonal_spatial_verification",
        help="Make seasonal spatial verification figures",
        action="store_true",
    )
    parser.add_argument(
        "-msv",
        "--do_monthly_spatial_verification",
        help="Make monthly spatial verification figures",
        action="store_true",
    )
    parser.add_argument(
        "-r", "--rsync", help="rsync figures to webserver", action="store_true"
    )

    args = parser.parse_args()

    if not (args.preds_labs_pickle):
        try:
            assert os.path.isdir(args.outdir[0])
            outdir = args.outdir[0]
        except (AssertionError, TypeError):
            print("You must provide an outdir if no preds_labs_pickle.")
            sys.exit(1)
        # If we don't provide a preds_labs_pickle, then we assume
        # that the data has already been parsed and we are making images
        # for things like spatial, monthly, or hourly verification.
        # Thus, we need an outdir.
        else:
            stride = args.stride[0]
    else:
        try:
            assert os.path.isfile(args.georeference_file[0])
        except (TypeError, AssertionError):
            print(
                "You must provide a georeference file if you want to parse out preds_labs_pickle."
            )
            sys.exit(1)
        # The georeference file is used for removing bad-satzen regions and spatial_verification.

    if args.do_seasonal_spatial_verification or args.do_monthly_spatial_verification:
        try:
            assert os.path.isfile(args.georeference_file[0])
        except (AssertionError, TypeError):
            print("You must provide a georeference file for this function.")
            sys.exit(1)
        # The georeference file is used for removing bad-satzen regions and spatial_verification.

    # solzen_thresh = 88
    satzen_thresh = 80

    if args.preds_labs_pickle:
        preds_labs_dict1 = pickle.load(open(args.preds_labs_pickle[0], "rb"))
        outdir = (
            os.path.dirname(args.preds_labs_pickle[0])
            if (not (args.outdir))
            else args.outdir[0]
        )
        utils.mkdir_p(outdir)
        all_preds = preds_labs_dict1["preds"]
        all_labels = preds_labs_dict1["labels"]
        datetimes = preds_labs_dict1["datetimes"]
        stride = preds_labs_dict1["stride"]

        # This is a one-off function to eliminate some bad GLM data in GOES-18 datetimes.
        # We're going to "eliminate" the data of datetimes that don't occur in GOES-17 list
        # but did occur in GOES-18 list.
        #    g17_data = pickle.load(open('/ships19/grain/probsevere/LC/g17g18/goes17/all_preds_labs.pkl','rb'))
        #    g17_dts = g17_data['datetimes']
        #    g17_dts = [x.replace(second=0,microsecond=0) for x in g17_dts]
        #    datetimes = [x.replace(second=0,microsecond=0) for x in datetimes]
        #    g18_dt_cter = 0
        #    init = len(datetimes)
        #    dts_to_pop = []
        #    print(init)
        #    for ii in range(all_preds.shape[0]):
        #      if(all_preds[ii].max() > 0): #valid g18 prediction time
        #        if(datetimes[g18_dt_cter] not in g17_dts):
        #          #zero-out data
        #          all_preds[ii] *= 0
        #          all_labels[ii] *= 0
        #          init -= 1
        #          dts_to_pop.append(datetimes[g18_dt_cter])
        #        g18_dt_cter += 1
        #    print(init)
        #    datetimes = [i for i in datetimes if i not in dts_to_pop]
        #    sys.exit()

        if (
            "midptX" in preds_labs_dict1
        ):  # Indicates that this is bunch of patches. Usually a validation set.
            # midpt[X,Y] are the center points, in FGF (usually for GE or GW CONUS/PACUS)
            # stride[X,Y] are each midpt[X,Y] divided by the stride.
            # Get the lats, lons, and satzen.
            nc = netCDF4.Dataset(args.georeference_file[0], "r")
            gip = nc.variables["goes_imager_projection"]
            x = nc.variables["x"][:]
            y = nc.variables["y"][:]
            projection = Proj(
                f"+proj=geos +lon_0={gip.longitude_of_projection_origin}"
                + f" +h={gip.perspective_point_height} +x_0=0.0 +y_0=0.0 +a="
                + f"{gip.semi_major_axis} +b={gip.semi_minor_axis}"
            )
            x *= gip.perspective_point_height
            y *= gip.perspective_point_height
            x = x[::stride]
            y = y[::stride]
            xx, yy = np.meshgrid(x, y)
            lons2D, lats2D = projection(xx, yy, inverse=True)
            shape = lats2D.shape
            from lightningcast.sat_zen_angle import sat_zen_angle

            satzen2D = sat_zen_angle(
                xlat=lats2D, xlon=lons2D, satlon=gip.longitude_of_projection_origin
            )
            # We use strideX/Y instead of midptX/Y because strideX/Y ARE the strided midptX/Ys
            satzen1D = np.array(
                [
                    satzen2D[zz[0], zz[1]]
                    for zz in zip(
                        preds_labs_dict1["strideY"], preds_labs_dict1["strideX"]
                    )
                ]
            )
            lons1D = np.array(
                [
                    lons2D[zz[0], zz[1]]
                    for zz in zip(
                        preds_labs_dict1["strideY"], preds_labs_dict1["strideX"]
                    )
                ]
            )
            lats1D = np.array(
                [
                    lats2D[zz[0], zz[1]]
                    for zz in zip(
                        preds_labs_dict1["strideY"], preds_labs_dict1["strideX"]
                    )
                ]
            )

            # Remove high-satzen patches. This removes crazy predictions on the satellite limb,
            # and also prevents inf lats/lons in the spacelook regions.
            good_patch_ind = np.where(
                satzen1D < satzen_thresh
            ) # where the satzen of the midpt is "good"
            print("SatZen:")
            print(satzen1D)
            all_good_preds = all_preds[good_patch_ind]
            all_good_labels = all_labels[good_patch_ind]
            good_dts = np.array(datetimes)[good_patch_ind]

            """
            for i in range(all_good_preds.shape[0]):
                plt.imshow(all_good_preds[i], cmap='binary')
                plt.colorbar()
                plt.savefig('pred'+str(i)+'.png')
                plt.close()
            """
            satzen1D = satzen1D[good_patch_ind]
            lons1D = lons1D[good_patch_ind]
            lats1D = lats1D[good_patch_ind]

            # Get local time
            tzf = TimezoneFinder()
            # This line is complicated. Essentially, we're turning UTC datetimes into local datetimes.
            # tzf.timezone_at(lng=zz[1], lat=zz[2]) #use lat and lon at each midpt to get timezone
            # pytz.timezone(<timezone>) #convert the string timezone into a timezone object that datetime can use
            # zz[0].replace(tzinfo=pytz.utc) #make the datetime timezone-aware by adding tzinfo=utc
            # <datetime obj>.astimezone(<timezone obj>) #convert from UTC to the local timezone.
            # It's in a list comprehension
            local_dts = [
                (zz[0].replace(tzinfo=pytz.utc)).astimezone(
                    pytz.timezone(tzf.timezone_at(lng=zz[1], lat=zz[2]))
                )
                for zz in zip(good_dts, lons1D, lats1D)
            ]

            good_preds = all_good_preds.flatten()
            good_labels = all_good_labels.flatten()
            # Some bulk verification for "ALL"
            print("Calc verification scores for ALL...")
            tmpdir = f"{outdir}/ALL/"
            utils.mkdir_p(tmpdir)
            scores_dict = verification(good_preds, good_labels, tmpdir)
            pickle.dump(scores_dict, open(f"{tmpdir}/scores_dict.pkl", "wb"))

            # month of each sample
            months = np.array([dt.month for dt in local_dts])
            themonths = np.arange(1, 13, 1)

            # by month
            for mm in themonths:
                MM = str(int(mm)).zfill(2)
                good_patch_ind = np.where(months == mm)
                if len(good_patch_ind[0]) > 0:
                    good_patch_ind = good_patch_ind[0]  # remove tuple
                    good_preds = all_good_preds[good_patch_ind].flatten()
                    good_labels = all_good_labels[good_patch_ind].flatten()
                    npatches = len(good_patch_ind)
                    npreds = len(good_preds)
                    print(f"Calc verification scores for month={MM}...")
                    tmpdir = f"{outdir}/month{MM}/"
                    utils.mkdir_p(tmpdir)
                    scores_dict = verification(good_preds, good_labels, tmpdir)
                    scores_dict["npreds"] = npreds
                    pickle.dump(scores_dict, open(f"{tmpdir}/scores_dict.pkl", "wb"))

                    if args.do_spatial_verification:
                        # spatial verification
                        print(f"Calc spatial verification for month={MM}...")
                        thresholds = [
                            0.1,
                            0.2,
                            0.25,
                            0.3,
                            0.35,
                            0.4,
                            0.45,
                            0.5,
                            0.6,
                            0.7,
                        ]
                        # get halfsizes for possibly strided patches
                        sy, sx = all_good_preds[0, :, :].shape
                        hsy = sy // 2
                        hsx = sx // 2
                        print("sy, sx")
                        print(sy, sx)

                        all_hits = np.zeros(
                            (len(thresholds), shape[0], shape[1]), dtype=np.int32
                        )
                        print("all_hits.shape:")
                        print(all_hits.shape)
                        all_misses = np.copy(all_hits)
                        all_FAs = np.copy(all_hits)
                        MM = str(int(mm)).zfill(2)
                        for ii in good_patch_ind:
                            preds = all_good_preds[ii]
                            targets = all_good_labels[ii]
                            hits, misses, FAs = spatial_verification(
                                preds.reshape(sy, sx),
                                targets.reshape(sy, sx),
                                thresholds=thresholds,
                                target_thresh=0.1,
                            )
                            # Find where in the grid the counts go
                            Y = preds_labs_dict1["strideY"][ii]
                            X = preds_labs_dict1["strideX"][ii]
                            print(Y, X)
                            #all_hits[:, Y - hsy : Y + hsy, X - hsx : X + hsx] += hits
                            #all_misses[:, Y - hsy : Y + hsy, X - hsx : X + hsx] += misses
                            #all_FAs[:, Y - hsy : Y + hsy, X - hsx : X + hsx] += FAs
                            all_hits[:, Y : Y + sy, X : X + sx] += hits
                            all_misses[:, Y : Y + sy, X : X + sx] += misses
                            all_FAs[:, Y : Y + sy, X : X + sx] += FAs
                        datasets = {}
                        global_atts = {}
                        dims = {"t": len(thresholds), "y": shape[0], "x": shape[1]}
                        datasets["hits"] = {
                            "data": all_hits,
                            "dims": ("t", "y", "x"),
                            "atts": {},
                        }
                        datasets["misses"] = {
                            "data": all_misses,
                            "dims": ("t", "y", "x"),
                            "atts": {},
                        }
                        datasets["FAs"] = {
                            "data": all_FAs,
                            "dims": ("t", "y", "x"),
                            "atts": {},
                        }
                        datasets["thresholds"] = {
                            "data": np.array(thresholds, dtype=float),
                            "dims": ("t"),
                            "atts": {},
                        }
                        global_atts[
                            "sample_shape"
                        ] = f"({npatches},{shape[0]},{shape[1]})"
                        global_atts["created"] = datetime.utcnow().strftime(
                            "%Y%m%d-%H%M UTC"
                        )
                        utils.write_netcdf(
                            f"{tmpdir}/spatial_counts_month{MM}.nc",
                            datasets,
                            dims,
                            atts=global_atts,
                            gzip=False,
                        )
                        print("NC:")
                        print(f"{tmpdir}/spatial_counts_month{MM}.nc")

            # by hour
            hours = np.array([dt.hour for dt in local_dts])
            thehours = np.arange(24)
            for hh in thehours:
                HH = str(int(hh)).zfill(2)
                good_patch_ind = np.where(hours == hh)
                if len(good_patch_ind[0]) > 0:
                    good_patch_ind = good_patch_ind[0]  # remove tuple
                    good_preds = all_good_preds[good_patch_ind].flatten()
                    good_labels = all_good_labels[good_patch_ind].flatten()
                    npatches = len(good_patch_ind)
                    npreds = len(good_preds)
                    print(f"Calc verification scores for local hour={HH}...")
                    tmpdir = f"{outdir}/hour{HH}/"
                    utils.mkdir_p(tmpdir)
                    scores_dict = verification(good_preds, good_labels, tmpdir)
                    scores_dict["npreds"] = npreds
                    pickle.dump(scores_dict, open(f"{tmpdir}/scores_dict.pkl", "wb"))

        else:  # indicates that every scene has the same geospatial extent

            nc = netCDF4.Dataset(args.georeference_file[0], "r")
            gip = nc.variables["goes_imager_projection"]
            x = nc.variables["x"][:]
            y = nc.variables["y"][:]
            projection = Proj(
                f"+proj=geos +lon_0={gip.longitude_of_projection_origin}"
                + f" +h={gip.perspective_point_height} +x_0=0.0 +y_0=0.0 +a="
                + f"{gip.semi_major_axis} +b={gip.semi_minor_axis}"
            )
            x *= gip.perspective_point_height
            y *= gip.perspective_point_height
            x = x[::stride]
            y = y[::stride]
            xx, yy = np.meshgrid(x, y)
            lons2D, lats2D = projection(xx, yy, inverse=True)
            shape = lats2D.shape
            from lightningcast.sat_zen_angle import sat_zen_angle

            satzen2D = sat_zen_angle(
                xlat=lats2D, xlon=lons2D, satlon=gip.longitude_of_projection_origin
            )

            # Go through array and find slices that have all 0s. These are frames
            # that didn't get filled b/c of FPT > threshold. Or we didn't have the fed_accum_data.
            good_ind = []
            for ii in range(all_preds.shape[0]):
                if all_preds[ii].max() > 0:
                    good_ind.append(ii)
            if len(good_ind):
                good_ind = np.array(good_ind)
                all_preds = all_preds[good_ind]
                all_labels = all_labels[good_ind]
                # save data back out?

            assert len(datetimes) == all_preds.shape[0] == all_labels.shape[0]

            # screen out bad satzen pixels
            bad_satzen_ind = np.where(satzen2D > satzen_thresh)
            all_preds[:, bad_satzen_ind[0], bad_satzen_ind[1]] = 0
            all_labels[:, bad_satzen_ind[0], bad_satzen_ind[1]] = 0

            if args.do_hourly_verification:
                # We do this here since need to retain the spatial coherence before reducing RAM.

                # We have a list of datetimes, 1 datetime per 2D sample. We need a LST at every point.
                # However, this is very expensive to compute, so we'll have to subsample the points.
                # Essentially, we say that a given LST dt is representative of a small patch/region.
                tzf = TimezoneFinder()
                patch_stride = ps = 128
                LSTs = np.zeros(all_preds.shape, dtype=np.int32) - 999
                for ii in range(LSTs.shape[0]):
                    for yy in range(ps // 2 // stride, LSTs.shape[1], ps // stride):
                        for xx in range(ps // 2 // stride, LSTs.shape[2], ps // stride):
                            if math.isinf(lons2D[yy, xx]) or math.isnan(lons2D[yy, xx]):
                                continue
                            LSTs[
                                ii,
                                yy - ps // 2 // stride : yy + ps // 2 // stride,
                                xx - ps // 2 // stride : xx + ps // 2 // stride,
                            ] = (
                                datetimes[ii]
                                .replace(tzinfo=pytz.utc)
                                .astimezone(
                                    pytz.timezone(
                                        tzf.timezone_at(
                                            lng=lons2D[yy, xx], lat=lats2D[yy, xx]
                                        )
                                    )
                                )
                            ).hour

            # Screen out very easy predictions to reduce RAM. These are predictions with P<0.01 and no lightning.
            ind = (all_preds >= 0.01) | (all_labels >= 1)
            all_preds = all_preds[ind]
            all_labels = all_labels[ind]
            if args.do_hourly_verification:
                LSTs = LSTs[ind]

            # performance for all predictions
            good_preds = all_preds.flatten()
            good_labels = all_labels.flatten()
            # Some bulk verification for "ALL"
            print("Calc verification scores for ALL...")
            tmpdir = f"{outdir}/ALL/"
            utils.mkdir_p(tmpdir)
            scores_dict = verification(
                good_preds, good_labels, tmpdir, climo=0.01
            )  # climo hard-coded
            pickle.dump(scores_dict, open(f"{tmpdir}/scores_dict.pkl", "wb"))

            # month of each sample
            months = np.array([dt.month for dt in datetimes])
            themonths = np.arange(1, 13, 1)

            # by month
            for mm in themonths:
                MM = str(int(mm)).zfill(2)
                good_patch_ind = np.where(months == mm)
                if len(good_patch_ind[0]) > 0:
                    good_patch_ind = good_patch_ind[0]  # remove tuple
                    good_preds = all_preds[good_patch_ind].flatten()
                    good_labels = all_labels[good_patch_ind].flatten()
                    npatches = len(good_patch_ind)
                    npreds = len(good_preds)
                    print(f"Calc verification scores for month={MM}...")
                    tmpdir = f"{outdir}/month{MM}/"
                    utils.mkdir_p(tmpdir)
                    scores_dict = verification(good_preds, good_labels, tmpdir)
                    scores_dict["npreds"] = npreds
                    pickle.dump(scores_dict, open(f"{tmpdir}/scores_dict.pkl", "wb"))

                    if args.do_spatial_verification:
                        # spatial verification
                        print(f"Calc spatial verification for month={MM}...")
                        thresholds = [
                            0.2,
                            0.25,
                            0.3,
                            0.35,
                            0.4,
                            0.45,
                            0.5,
                            0.6,
                            0.7,
                            0.9,
                        ]
                        # get halfsizes for possibly strided patches
                        sy, sx = all_preds[0, :, :].shape
                        hsy = sy // 2
                        hsx = sx // 2

                        all_hits = np.zeros((len(thresholds), sy, sx), dtype=np.int32)
                        all_misses = np.copy(all_hits)
                        all_FAs = np.copy(all_hits)
                        MM = str(int(mm)).zfill(2)
                        for ii in good_patch_ind:
                            preds = all_preds[ii]
                            targets = all_labels[ii]
                            hits, misses, FAs = spatial_verification(
                                preds.reshape(sy, sx),
                                targets.reshape(sy, sx),
                                thresholds=thresholds,
                                target_thresh=0.1,
                            )
                            all_hits += hits
                            all_misses += misses
                            all_FAs += FAs
                        datasets = {}
                        global_atts = {}
                        dims = {"t": len(thresholds), "y": shape[0], "x": shape[1]}
                        datasets["hits"] = {
                            "data": all_hits,
                            "dims": ("t", "y", "x"),
                            "atts": {},
                        }
                        datasets["misses"] = {
                            "data": all_misses,
                            "dims": ("t", "y", "x"),
                            "atts": {},
                        }
                        datasets["FAs"] = {
                            "data": all_FAs,
                            "dims": ("t", "y", "x"),
                            "atts": {},
                        }
                        datasets["thresholds"] = {
                            "data": np.array(thresholds, dtype=float),
                            "dims": ("t"),
                            "atts": {},
                        }
                        global_atts["sample_shape"] = f"({npatches},{sy},{sx})"
                        global_atts["created"] = datetime.utcnow().strftime(
                            "%Y%m%d-%H%M UTC"
                        )
                        utils.write_netcdf(
                            f"{tmpdir}/spatial_counts_month{MM}.nc",
                            datasets,
                            dims,
                            atts=global_atts,
                            gzip=False,
                        )

            # by hour
            thehours = np.arange(24)
            for hh in thehours:
                HH = str(int(hh)).zfill(2)
                good_ind = np.where(LSTs == hh)
                if len(good_ind[0]) > 0:
                    good_preds = all_preds[good_ind]
                    good_labels = all_labels[good_ind]
                    npreds = len(good_preds)
                    print(f"Calc verification scores for local hour={HH}...")
                    tmpdir = f"{outdir}/hour{HH}/"
                    utils.mkdir_p(tmpdir)
                    scores_dict = verification(good_preds, good_labels, tmpdir)
                    scores_dict["npreds"] = npreds
                    pickle.dump(scores_dict, open(f"{tmpdir}/scores_dict.pkl", "wb"))

    # now make the figures

    if args.do_monthly_verification:
        # assumes that monthly data has already been parsed in outdir
        parsed_perfdiagram(outdir, figtype="month")
        parsed_reliability(outdir, figtype="month")
        scalars_timeseries(outdir, figtype="month")

    if args.do_hourly_verification:
        # assumes that hourly data has already been parsed in outdir
        parsed_perfdiagram(outdir, figtype="hour")
        parsed_reliability(outdir, figtype="hour")
        scalars_timeseries(outdir, figtype="hour")

    # Set up
    webdir = "/webdata/web/cimss/htdocs/severe_conv/LC/tests/"
    webuser = "jcintineo@webaccess"
    if (
        os.path.basename(os.path.dirname(outdir)) == "TEST"
    ):  # default outdir from tf_eval.py
        test = os.path.basename(os.path.dirname(os.path.dirname(outdir)))
    else:  # user-supplied
        if outdir[-1] == "/":
            outdir = outdir[0:-1]  # remove /
        test = os.path.basename(outdir)

    # Make montage_stats.png
    if args.do_monthly_verification and args.do_hourly_verification:
        os.system(
            f"montage {outdir}/ALL/performance_diagram.png {outdir}/ALL/attributes_diagram.png xc:white "
            + f"{outdir}/month_performance.png {outdir}/month_reliability.png {outdir}/month_scalars.png "
            + f"{outdir}/hour_performance.png {outdir}/hour_reliability.png {outdir}/hour_scalars.png -border 1%x1% -geometry 500x500+2+2 "
            + f"-tile 3x3 -frame 0 {outdir}/montage_stats.png"
        )
        if args.rsync:
            os.system(f"ssh {webuser} 'mkdir -p {webdir}/{test}'")
            os.system(
                f"rsync -ltD {outdir}/montage_stats.png {webuser}:{webdir}/{test}/"
            )

    if args.do_monthly_spatial_verification:
        monthly_spatial_verification(
            outdir,
            problevel=0.3,
            georeference_file=args.georeference_file[0],
            stride=stride,
        )  # FIXME hard-coded prob-level?

    if args.do_seasonal_spatial_verification:
        #seasons = ["DJF", "MAM", "JJA", "SON", "ALL"]
        seasons = ['ALL']
        seasonal_spatial_verification(
            outdir,
            problevel=0.35,
            georeference_file=args.georeference_file[0],  # FIXME hard-coded prob-level?
            stride=stride,
            seasons=seasons,
        )
        # Make montages
        rsync_cmd = ""
        for ii, season in enumerate(seasons):
            os.system(
                f"montage {outdir}/{season}/geo_pod.png {outdir}/{season}/geo_far.png "
                + f"{outdir}/{season}/geo_csi.png {outdir}/{season}/obs_cts.png -border 1%x1% "
                + f"-geometry 400x400+2+2 -tile 2x2 -frame 0 {outdir}/{season}/geo_montage.png"
            )
            os.system(
                f"montage {outdir}/{season}/geo_pod.png {outdir}/{season}/geo_far.png "
                + f"{outdir}/{season}/geo_csi.png {outdir}/{season}/obs_cts.png -border 1%x1% "
                + f"-geometry 400x400+2+2 -tile 2x2 -frame 0 {outdir}/{season}/geo_montage.eps"
            )
            if os.path.isfile(f"{outdir}/{season}/geo_montage.png"):
                if args.rsync:
                    if rsync_cmd == "":
                        rsync_cmd += "rsync -ltD "
                    rsync_cmd += f"{outdir}/{season}/geo_montage.png "

        # Send montages to webserver, if successfully created
        if rsync_cmd and args.rsync:
            os.system(f"ssh {webuser} 'mkdir -p {webdir}/{test}'")
            os.system(f"{rsync_cmd} {webuser}:{webdir}/{test}/")
