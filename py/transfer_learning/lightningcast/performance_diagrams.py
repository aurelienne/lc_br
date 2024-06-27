"""Methods for plotting performance diagram."""

import inspect, os, sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(current_dir))

import glob
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from lightningcast import bootstrap
import time

DEFAULT_LINE_COLOUR = np.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 2
DEFAULT_BIAS_LINE_COLOUR = np.full(3, 152.0 / 255)
DEFAULT_BIAS_LINE_WIDTH = 1

LEVELS_FOR_CSI_CONTOURS = np.linspace(0, 1, num=11, dtype=float)
LEVELS_FOR_BIAS_CONTOURS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])

BIAS_STRING_FORMAT = "%.2f"
BIAS_LABEL_PADDING_PX = 10

FIGURE_WIDTH_INCHES = 4
FIGURE_HEIGHT_INCHES = 4

FONT_SIZE = 8
plt.rc("font", size=FONT_SIZE)
plt.rc("axes", titlesize=FONT_SIZE)
plt.rc("axes", labelsize=FONT_SIZE)
plt.rc("xtick", labelsize=FONT_SIZE)
plt.rc("ytick", labelsize=FONT_SIZE)
plt.rc("legend", fontsize=FONT_SIZE)
plt.rc("figure", titlesize=FONT_SIZE)

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


def _get_sr_pod_grid(success_ratio_spacing=0.01, pod_spacing=0.01):
    """Creates grid in SR-POD (success ratio / probability of detection) space.
    M = number of rows (unique POD values) in grid
    N = number of columns (unique success ratios) in grid
    :param success_ratio_spacing: Spacing between grid cells in adjacent
        columns.
    :param pod_spacing: Spacing between grid cells in adjacent rows.
    :return: success_ratio_matrix: M-by-N np array of success ratios.
        Success ratio increases with column index.
    :return: pod_matrix: M-by-N np array of POD values.  POD decreases with
        row index.
    """

    num_success_ratios = 1 + int(np.ceil(1.0 / success_ratio_spacing))
    num_pod_values = 1 + int(np.ceil(1.0 / pod_spacing))

    unique_success_ratios = np.linspace(0.0, 1.0, num=num_success_ratios)
    unique_pod_values = np.linspace(0.0, 1.0, num=num_pod_values)[::-1]
    return np.meshgrid(unique_success_ratios, unique_pod_values)


def _csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: csi_array: np array (same shape) of CSI values.
    """

    return (success_ratio_array**-1 + pod_array**-1 - 1.0) ** -1


def _bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: frequency_bias_array: np array (same shape) of frequency biases.
    """

    return pod_array / success_ratio_array


def _get_csi_colour_scheme():
    """Returns colour scheme for CSI (critical success index).
    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = plt.cm.Blues
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(
        this_colour_norm_object(LEVELS_FOR_CSI_CONTOURS)
    )
    colour_list = [rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(np.array([1, 1, 1]))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _add_colour_bar(
    axes_object,
    colour_map_object,
    values_to_colour,
    min_colour_value,
    max_colour_value,
    colour_norm_object=None,
    orientation_string="vertical",
    extend_min=True,
    extend_max=True,
    fraction_of_axis_length=1.0,
    font_size=FONT_SIZE,
):
    """Adds colour bar to existing axes.
    :param axes_object: Existing axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.plt.cm`).
    :param values_to_colour: np array of values to colour.
    :param min_colour_value: Minimum value in colour map.
    :param max_colour_value: Max value in colour map.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.  If `colour_norm_object is None`,
        will assume that scale is linear.
    :param orientation_string: Orientation of colour bar ("vertical" or
        "horizontal").
    :param extend_min: Boolean flag.  If True, the bottom of the colour bar will
        have an arrow.  If False, it will be a flat line, suggesting that lower
        values are not possible.
    :param extend_max: Same but for top of colour bar.
    :param fraction_of_axis_length: Fraction of axis length (y-axis if
        orientation is "vertical", x-axis if orientation is "horizontal")
        occupied by colour bar.
    :param font_size: Font size for labels on colour bar.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.plt.colorbar`) created by this method.
    """

    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False
        )

    scalar_mappable_object = plt.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = "both"
    elif extend_min:
        extend_string = "min"
    elif extend_max:
        extend_string = "max"
    else:
        extend_string = "neither"

    if orientation_string == "horizontal":
        padding = 0.075
    else:
        padding = 0.05

    colour_bar_object = plt.colorbar(
        ax=axes_object,
        mappable=scalar_mappable_object,
        orientation=orientation_string,
        pad=padding,
        extend=extend_string,
        shrink=fraction_of_axis_length,
    )

    colour_bar_object.ax.tick_params(labelsize=font_size)
    return colour_bar_object


def _get_points_in_perf_diagram(observed_labels, forecast_probabilities, nboots=0):
    """Creates points for performance diagram.
    E = number of examples
    T = number of binarization thresholds
    :param observed_labels: length-E np array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E np array with forecast
        probabilities of label = 1.
    :return: pod_by_threshold: length-T np array of POD (probability of
        detection) values.
    :return: success_ratio_by_threshold: length-T np array of success ratios.
    """

    assert np.all(np.logical_or(observed_labels == 0, observed_labels == 1))

    assert np.all(
        np.logical_and(forecast_probabilities >= 0, forecast_probabilities <= 1)
    )

    observed_labels = observed_labels.astype(int)
    binarization_thresholds = np.linspace(0, 1, num=101, dtype=float)

    num_thresholds = len(binarization_thresholds)
    pod_by_threshold = np.full(num_thresholds, np.nan)
    csi_by_threshold = np.full(num_thresholds, np.nan)
    success_ratio_by_threshold = np.full(num_thresholds, np.nan)
    binary_accuracy_by_threshold = np.full(num_thresholds, np.nan)
    binary_freq_bias_by_threshold = np.full(num_thresholds, np.nan)
    pss_by_threshold = np.full(num_thresholds, np.nan)

    if nboots > 0:
        # tmp_thresh = np.arange(0,1.01,0.01)
        tmp_thresh = np.copy(binarization_thresholds)
        tmp_nthresh = len(tmp_thresh)
        pod_by_threshold_05 = np.full(tmp_nthresh, np.nan)
        pod_by_threshold_95 = np.full(tmp_nthresh, np.nan)
        success_ratio_by_threshold_05 = np.full(tmp_nthresh, np.nan)
        success_ratio_by_threshold_95 = np.full(tmp_nthresh, np.nan)
        sample_ind = np.arange(len(observed_labels), dtype=int)
        boot_ind = bootstrap.bootstrap(sample_ind, bootnum=nboots).astype(
            int
        )  # boot_ind has shape of nboots by len(observed_labels)
        all_pods = np.full((nboots, tmp_nthresh), np.nan)
        all_srs = np.full((nboots, tmp_nthresh), np.nan)

        for k in range(tmp_nthresh):
            for ii in range(nboots):  # each row in boot_ind
                tmp_fcsts = forecast_probabilities[boot_ind[ii]]
                tmp_obs = observed_labels[boot_ind[ii]]
                fcst_labs = (tmp_fcsts >= tmp_thresh[k]).astype(int)
                hits = np.sum(np.logical_and(fcst_labs == 1, tmp_obs == 1))
                misses = np.sum(np.logical_and(fcst_labs == 0, tmp_obs == 1))
                FAs = np.sum(np.logical_and(fcst_labs == 1, tmp_obs == 0))
                all_pods[ii, k] = float(hits) / (hits + misses)
                all_srs[ii, k] = float(hits) / (hits + FAs)

        pod05 = np.percentile(
            all_pods, 5, axis=0
        )  # shape of binarization_thresholds
        pod95 = np.percentile(
            all_pods, 95, axis=0
        )  # shape of binarization_thresholds
        sr05 = np.percentile(all_srs, 5, axis=0)  # shape of binarization_thresholds
        sr95 = np.percentile(all_srs, 95, axis=0)  # shape of binarization_thresholds

    for k in range(num_thresholds):
        # t0=time.time()
        these_forecast_labels = (
            forecast_probabilities >= binarization_thresholds[k]
        ).astype(int)

        this_num_hits = np.sum(
            np.logical_and(these_forecast_labels == 1, observed_labels == 1)
        )

        this_num_false_alarms = np.sum(
            np.logical_and(these_forecast_labels == 1, observed_labels == 0)
        )

        this_num_misses = np.sum(
            np.logical_and(these_forecast_labels == 0, observed_labels == 1)
        )

        this_num_cns = np.sum(
            np.logical_and(these_forecast_labels == 0, observed_labels == 0)
        )

        try:
            pod_by_threshold[k] = float(this_num_hits) / (
                this_num_hits + this_num_misses
            )
        except ZeroDivisionError:
            pass

        try:
            success_ratio_by_threshold[k] = float(this_num_hits) / (
                this_num_hits + this_num_false_alarms
            )
        except ZeroDivisionError:
            pass

        try:
            csi_by_threshold[k] = float(this_num_hits) / (
                this_num_hits + this_num_misses + this_num_false_alarms
            )
        except ZeroDivisionError:
            pass

        try:
            binary_accuracy_by_threshold[k] = (float(this_num_hits) + this_num_cns) / (
                this_num_hits + this_num_misses + this_num_false_alarms + this_num_cns
            )
        except ZeroDivisionError:
            pass

        try:
            binary_freq_bias_by_threshold[k] = (
                float(this_num_hits) + this_num_false_alarms
            ) / (this_num_hits + this_num_misses)
        except ZeroDivisionError:
            pass

        try:
            pss_by_threshold[k] = (
                (this_num_hits * this_num_cns)
                - (this_num_false_alarms * this_num_misses)
            ) / (
                (this_num_hits + this_num_misses)
                * (this_num_false_alarms + this_num_cns)
            )
        except ZeroDivisionError:
            pass
        # print(k,time.time() - t0)

    pod_by_threshold = np.array([1.0] + pod_by_threshold.tolist() + [0.0])
    success_ratio_by_threshold = np.array(
        [0.0] + success_ratio_by_threshold.tolist() + [1.0]
    )

    if nboots > 0:
        sr95 = np.array([0.0] + sr95.tolist() + [1.0])
        sr05 = np.array([0.0] + sr05.tolist() + [1.0])
        pod95 = np.array([1.0] + pod95.tolist() + [0.0])
        pod05 = np.array([1.0] + pod05.tolist() + [0.0])

        import pandas as pd

        tmp = pd.DataFrame(columns=["pod05", "pod", "pod95", "sr05", "sr", "sr95"])
        tmp["pod05"] = pod05
        tmp["pod"] = pod_by_threshold
        tmp["pod95"] = pod95
        tmp["sr05"] = sr05
        tmp["sr"] = success_ratio_by_threshold
        tmp["sr95"] = sr95

        print(tmp[50:55])

        return {
            "pod": pod_by_threshold,
            "sr": success_ratio_by_threshold,
            "csi": csi_by_threshold,
            "bin_acc": binary_accuracy_by_threshold,
            "bin_bias": binary_freq_bias_by_threshold,
            "pss": pss_by_threshold,
            "pod05": pod05,
            "pod95": pod95,
            "sr05": sr05,
            "sr95": sr95,
        }
    else:
        return {
            "pod": pod_by_threshold,
            "sr": success_ratio_by_threshold,
            "csi": csi_by_threshold,
            "bin_acc": binary_accuracy_by_threshold,
            "bin_bias": binary_freq_bias_by_threshold,
            "pss": pss_by_threshold,
        }


def plot_performance_diagram(
    observed_labels,
    forecast_probabilities,
    line_colour=DEFAULT_LINE_COLOUR,
    line_width=DEFAULT_LINE_WIDTH,
    bias_line_colour=DEFAULT_BIAS_LINE_COLOUR,
    bias_line_width=DEFAULT_BIAS_LINE_WIDTH,
    nboots=0,
    return_axes=False,
):
    """Plots performance diagram.
    E = number of examples
    :param observed_labels: length-E np array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E np array with forecast
        probabilities of label = 1.
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param bias_line_colour: Colour of contour lines for frequency bias.
    :param bias_line_width: Width of contour lines for frequency bias.
    :param nboots: number of bootstaps for CI (default=0).
    :return: pod_by_threshold: See doc for `_get_points_in_perf_diagram`.
        detection) values.
    :return: success_ratio_by_threshold: Same.
    """

    scores_dict = _get_points_in_perf_diagram(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities,
        nboots=nboots,
    )

    # import pickle
    # scores_dict = pickle.load(open('/ships19/grain/jcintineo/GLM/goes_east/FD/202001-12/validation4/ALL/scores_dict.pkl','rb'))

    pod_by_threshold = scores_dict["pod"]  # [:-2]
    success_ratio_by_threshold = scores_dict["sr"]  # [:-2]

    _, axes_object = plt.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    success_ratio_matrix, pod_matrix = _get_sr_pod_grid()
    csi_matrix = _csi_from_sr_and_pod(success_ratio_matrix, pod_matrix)
    frequency_bias_matrix = _bias_from_sr_and_pod(success_ratio_matrix, pod_matrix)

    this_colour_map_object, this_colour_norm_object = _get_csi_colour_scheme()

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

    colour_bar_object = _add_colour_bar(
        axes_object=axes_object,
        colour_map_object=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        values_to_colour=csi_matrix,
        min_colour_value=0.0,
        max_colour_value=1.0,
        orientation_string="vertical",
        extend_min=False,
        extend_max=False,
    )
    colour_bar_object.set_label("CSI (critical success index)")

    bias_colour_tuple = ()
    for _ in range(len(LEVELS_FOR_BIAS_CONTOURS)):
        bias_colour_tuple += (bias_line_colour,)

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
        fontsize=FONT_SIZE,
    )

    nan_flags = np.logical_or(
        np.isnan(success_ratio_by_threshold), np.isnan(pod_by_threshold)
    )

    if not np.all(nan_flags):
        real_indices = np.where(np.invert(nan_flags))[0]
        axes_object.plot(
            success_ratio_by_threshold[real_indices],
            pod_by_threshold[real_indices],
            color=line_colour,
            linestyle="solid",
            linewidth=line_width,
        )

        if nboots > 0:
            tmp_sr_by_thresh = np.copy(success_ratio_by_threshold)
            # print(len(tmp_sr_by_thresh),len(scores_dict['pod05']),len(scores_dict['pod95']))
            # tmp_sr_by_thresh = success_ratio_by_threshold[0::10]  #only works if tmp_thresh = np.arange(0,1.01,0.01) and thresh=np.linspace(0,1,num=1001)
            axes_object.fill_between(
                tmp_sr_by_thresh,
                scores_dict["pod05"],
                scores_dict["pod95"],
                facecolor=line_colour,
                alpha=0.3,
            )

        xs = (
            [success_ratio_by_threshold[5]]
            + list(success_ratio_by_threshold[10:100:10])
            + [success_ratio_by_threshold[95]]
        )
        ys = (
            [pod_by_threshold[5]]
            + list(pod_by_threshold[10:100:10])
            + [pod_by_threshold[95]]
        )
        labs = ["5", "10", "20", "30", "40", "50", "60", "70", "80", "90", "95"]
        axes_object.plot(
            xs, ys, linestyle="None", color=line_colour, marker="o", markersize=6
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

    axes_object.set_xlabel("Success ratio (1 - FAR)")
    axes_object.set_ylabel("POD (probability of detection)")
    axes_object.set_xlim(0.0, 1.0)
    axes_object.set_ylim(0.0, 1.0)

    if return_axes:
        return scores_dict, axes_object
    else:
        return scores_dict


def perf_diagram_pod_far(pod, far, outfilename, labs = [0,10,20,30,40,50,60,70,80,90],
                 color=np.array([228, 26, 28], dtype=float) / 255,
                 linewidth=5,
                 pod2=None,far2=None,labs2=None,color2=np.array([255, 181, 43], dtype=float) / 255,
                 legend_labs=['v1','v2']):
    '''
    plots performance diagram, but with already-calculated POD/FAR points
    - pod: array of pods
    - far: array of fars
    - labs: the labels for the discrete bins
    - outfilename: string
    '''

    assert len(labs) == len(pod) == len(far)
 
    FONT_SIZE = 16
    plt.rc('font', size=FONT_SIZE)
    plt.rc('axes', titlesize=FONT_SIZE)
    plt.rc('axes', labelsize=FONT_SIZE)
    plt.rc('xtick', labelsize=FONT_SIZE)
    plt.rc('ytick', labelsize=FONT_SIZE)
    plt.rc('legend', fontsize=FONT_SIZE)
    plt.rc('figure', titlesize=FONT_SIZE)
  
    _, axes_object = plt.subplots(
          1, 1, figsize=(8, 8)
      )
  
    pod = np.array(pod)
    far = np.array(far)
  
    LEVELS_FOR_CSI_CONTOURS = np.linspace(0, 1, num=11, dtype=float)
    LEVELS_FOR_BIAS_CONTOURS = np.array(
      [0.25, 0.5, 0.75, 1., 1.5, 2., 3., 5.])
    bias_line_colour = np.full(3, 152. / 255)
    BIAS_STRING_FORMAT = '%.2f'
    BIAS_LABEL_PADDING_PX = 10
  
    success_ratio_matrix, pod_matrix = _get_sr_pod_grid()
    csi_matrix = _csi_from_sr_and_pod(success_ratio_matrix, pod_matrix)
    frequency_bias_matrix = _bias_from_sr_and_pod(
          success_ratio_matrix, pod_matrix)
  
    this_colour_map_object, this_colour_norm_object = _get_csi_colour_scheme()
  
    plt.contourf(
        success_ratio_matrix, pod_matrix, csi_matrix, LEVELS_FOR_CSI_CONTOURS,
        cmap=this_colour_map_object, norm=this_colour_norm_object, vmin=0.,
        vmax=1., axes=axes_object)
  
    colour_bar_object = _add_colour_bar(
        axes_object=axes_object, colour_map_object=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        values_to_colour=csi_matrix, min_colour_value=0.,
        max_colour_value=1., orientation_string='vertical',
        extend_min=False, extend_max=False, font_size=FONT_SIZE)
    colour_bar_object.set_label('CSI (critical success index)',fontsize=FONT_SIZE)
  
    bias_colour_tuple = ()
    for _ in range(len(LEVELS_FOR_BIAS_CONTOURS)):
        bias_colour_tuple += (bias_line_colour,)
  
    bias_contour_object = plt.contour(
        success_ratio_matrix, pod_matrix, frequency_bias_matrix,
        LEVELS_FOR_BIAS_CONTOURS, colors=bias_colour_tuple,
        linewidths=1, linestyles='dashed', axes=axes_object)
    plt.clabel(
        bias_contour_object, inline=True, inline_spacing=BIAS_LABEL_PADDING_PX,
        fmt=BIAS_STRING_FORMAT, fontsize=FONT_SIZE-2)
  
    if(pod2 is not None): #plot secondary line first
      pod2 = np.array(pod2); far2 = np.array(far2)
      success_ratio2 = 1 - far2
      csi2Line, = axes_object.plot(success_ratio2, pod2, color=color2, linestyle='solid',linewidth=linewidth)
  
      xs = list(success_ratio2)
      ys = list(pod2)
      labs2 = [str(_) for _ in labs2]
      axes_object.plot(xs, ys, color=color2, markeredgecolor=color2, marker='o', linestyle='None', markersize=8)
  
      #labs, xs, and ys need to be same len
      assert(len(labs2) == len(xs) == len(ys))
      for i in range(0,len(labs2)):
        #if(int(labs[i])<10):
        #  xcor = 0.01; ycor = 0.0075
        #else:
        #  xcor = 0.02; ycor = 0.0075
        #axes_object.annotate(labs2[i], xy=(xs[i]-xcor, ys[i]-ycor), color='white', fontsize=10)
        axes_object.annotate(labs2[i], xy=(xs[i]-0.04,ys[i]), color='black', fontsize=8)
  
  
    success_ratio = 1 - far
    csi1Line, = axes_object.plot(
        success_ratio,pod, color=color,
        linestyle='solid', linewidth=linewidth)
  
    xs = list(success_ratio)
    ys = list(pod)
    labs = [str(_) for _ in labs]
    axes_object.plot(xs, ys, color=color, markeredgecolor=color, marker='o', linestyle='None', markersize=8)
  
    #labs, xs, and ys need to be same len
    assert(len(labs) == len(xs) == len(ys))
    for i in range(0,len(labs)): 
      #if(int(labs[i])<10):
      #  xcor = 0.01; ycor = 0.0075
      #else:
      #  xcor = 0.02; ycor = 0.0075
      #axes_object.annotate(labs[i], xy=(xs[i]-xcor, ys[i]-ycor), color='white', fontsize=10)
      axes_object.annotate(labs[i], xy=(xs[i]+0.02,ys[i]), color='black', fontsize=8)  


    axes_object.set_xlabel('Success ratio (1 - FAR)',fontsize=FONT_SIZE)
    axes_object.set_ylabel('POD (probability of detection)',fontsize=FONT_SIZE)
    axes_object.tick_params('x',labelsize=FONT_SIZE)
    axes_object.tick_params('y',labelsize=FONT_SIZE)
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)
  
    if(pod2 is not None):
      leg = axes_object.legend([csi1Line,csi2Line], legend_labs,loc='upper right',fontsize=18)
  
    plt.savefig(outfilename,bbox_inches="tight")


def perf_diagram_pkls(pkl1, outdir, pkl2=None, legend_labs=['v1','v2'], index=[0,1]):

    """
    Runs perf_diagram_pod_far with contents of one or two pkl files (full paths).
    Assumes keys in the pickle such as 'pod10_index0', 'far35_index0', etc., where the number is 
    the binarized probability threshold. index0 or index1 indicates the index of the 1st
    or second output. Default is to make diagrams for first 2 outputs. If pkl2 is supplied
    and index=[0,1], first figure will have index0 outputs for pkl1 and pkl2, and second figure
    will have index1 outputs for pkl1 and pkl2. 
    """

    print(pkl1)
    if pkl2 and len(index) == 2:
        assert len(legend_labs) == 4

    if isinstance(index, int):
        index = [index]
    elif isinstance(index, list):
        for _ in index:
            assert(isinstance(_, int))
    else:
        print("index must be an int or a list of ints")
        sys.exit(1)

    scores1 = pickle.load(open(pkl1,'rb'))

    for ii in index:

        labels = []
        labels2 = []
        pod1 = []
        far1 = []
        pod2 = None
        far2 = None
    
        for key, val in scores1.items():
            if 'pod' in key and f'index{ii}' in key:
                prob = key.split('_')[0][3:5]
                labels.append(int(prob))
                pod1.append(val)
                far1.append(scores1[f'far{prob}_index{ii}'])
        # Sort values in ascending order of probability labels
        pod1 = np.array(pod1)
        far1 = np.array(far1)
        labels = np.array(labels)
        pod1 = pod1[labels.argsort()]
        far1 = far1[labels.argsort()]
    
        if pkl2:
            scores2 = pickle.load(open(pkl2,'rb'))
    
            labels2 = []
            pod2 = []
            far2 = []
        
            for key, val in scores2.items():
                if 'pod' in key and f'index{ii}' in key:
                    prob = key.split('_')[0][-2:]
                    labels2.append(int(prob))
                    pod2.append(val)
                    far2.append(scores2[f'far{prob}_index{ii}'])
            # Sort values in ascending order of probability labels
            pod2 = np.array(pod2)
            far2 = np.array(far2)
            labels2 = np.array(labels2)
            pod2 = pod2[labels2.argsort()]
            far2 = far2[labels2.argsort()]
    
        # Assume legend_labs[0:2] are for the first figure and 
        # legend_labs[2:4] are for the second figure
        if ii == 0:
            ll = legend_labs[0:2]
        elif ii == 1:
            ll = legend_labs[2:4] 
            if len(ll) == 0:
                # Just repeat legend_labs[0:2] if addl legend_labs weren't supplied
                ll = legend_labs[0:2] 

        outfilename = f"{outdir}/perfdiagram_index{ii}"
        perf_diagram_pod_far(pod1, far1, outfilename, labs = labels,
                             pod2=pod2, far2=far2, labs2 = labels2, 
                             legend_labs=ll)

#---------------------------------------------------------------------------------------


def parsed_perfdiagram(outdir, figtype="month", index=0):
    '''
    For making one performance diagram with a curve for each month or hour.
    - outdir should contain directories like "month01-02", "month10", "hour22", etc.
    - index is for the ith model output. Usually there is only 1 (index=0) or 2 (index=1).
    '''

    assert figtype == "month" or figtype == "hour"

    cmap = plt.get_cmap("hsv")
    dirs = np.sort(glob.glob(f"{outdir}/{figtype}??") +
                   glob.glob(f"{outdir}/{figtype}??-??"))
    ndirs = len(dirs)

    figure_object, axes_object = plt.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    # get some set up stuff 
    success_ratio_matrix, pod_matrix = _get_sr_pod_grid()
    csi_matrix = _csi_from_sr_and_pod(
        success_ratio_matrix, pod_matrix
    )
    frequency_bias_matrix = _bias_from_sr_and_pod(
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

    colour_bar_object = _add_colour_bar(
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
        inds4colors = np.linspace(0, 1, ndirs)
        divisions = ndirs
    if figtype == "hour":
        inds4colors = np.linspace(0, 1, ndirs)
        divisions = ndirs
    for ii,thedir in enumerate(dirs):
        val_color = cmap(inds4colors[ii])

        scores = pickle.load(open(f"{thedir}/eval_results.pkl", "rb"))

        labels = []
        pod = []
        far = []

        for key, val in scores.items():
            if 'pod' in key and f'index{index}' in key:
                prob = key.split('_')[0][3:5]
                labels.append(int(prob))
                pod.append(val)
                far.append(scores[f'far{prob}_index{index}'])
        # Sort values in ascending order of probability labels
        pod = np.array(pod)
        far = np.array(far)
        labels = np.array(labels)
        pod = pod[labels.argsort()]
        far = far[labels.argsort()]

        axes_object.plot(1-far, pod, color=val_color)

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
        month_names = []
        for dd in dirs:
            month_names.append(get_month_name(dd.split('month')[1]))
        cbar.ax.set_xticklabels(month_names, rotation="vertical")
    elif figtype == "hour": #FIXME 24 hours assumed
        cbar.ax.set_xticklabels(
            [str(hh).zfill(2) for hh in np.arange(24, dtype=int)], rotation="vertical"
        )
    cbar.ax.tick_params(size=0, labelsize=10)
    plt.savefig(f"{outdir}/{figtype}_performance_index{index}.png", bbox_inches="tight", dpi=300)
    print(f"Saved {outdir}/{figtype}_performance_index{index}.png")
