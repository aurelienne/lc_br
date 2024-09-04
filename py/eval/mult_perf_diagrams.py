import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightningcast import performance_diagrams
from lightningcast import attributes_diagrams
import sys, os
import pickle
import numpy as np
import time
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import auc as skl_auc
import sklearn.metrics
from collections import defaultdict


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
    label,
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
            label=label
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
                label=label,
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
                label=label,
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

def verification(conv_preds, labels, outdir="", climo=-1):

    print(conv_preds[0].shape)
    print(labels[0].shape)
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
    scores_dict1 = performance_diagrams.plot_performance_diagram(
        observed_labels=labels[0],
        forecast_probabilities=conv_preds[0],
        label='Original',
        line_colour=color1,
        nboots=0,
    )
    if len(conv_preds) > 1:
        plot_line(
            plt.gca(), plt.gcf(), conv_preds[1], labels[1], label='Fine-Tuned (Full)', color=color2, perf=True
        )
    perf_diagram_file_name = os.path.join(outdir, "performance_diagram.png")
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig(perf_diagram_file_name, dpi=300, bbox_inches="tight")
    plt.close()

    lw = 1.5
    # Attributes Diagram
    (
        mean_forecast_probs,
        mean_event_frequencies,
        num_examples_by_bin,
        ax_ob,
        fig_ob,
    ) = attributes_diagrams.plot_attributes_diagram(
        observed_labels=labels[0],
        forecast_probabilities=conv_preds[0],
        label='Original',
        num_bins=20,
        return_main_axes=True,
        nboots=0,
        plot_hist=plot_hist,
        color=color1,
        climo=climo,
    )
    if len(conv_preds) > 1:
        plot_line(
            ax_ob, fig_ob, conv_preds[1], labels[1], label='Fine-Tuned (Full)', color=color2, atts=True, lw=lw
        )
        """
        # plot histogram
        attributes_diagrams._plot_forecast_histogram_same_axes(
            axes_object=ax_ob,
            num_examples_by_bin=num_examples_by_bin,
            facecolor=main_color,
        )
        """
    
    attr_diagram_file_name = os.path.join(outdir, "attributes_diagram.png")
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig(attr_diagram_file_name, dpi=300, bbox_inches="tight")
    plt.close()


def merge_pkl_files(pkl_list_file):
    dicts_list = []
    pkl_list = open(pkl_list_file, 'r')
    for pkl_file in pkl_list.readlines():
        pkl_dict = pickle.load(open(pkl_file.strip(), 'rb'))
        dicts_list.append(pkl_dict)

    all_dicts = {}
    i = 0
    for d in dicts_list:
        for key, values in d.items():
            if key == 'stride':
                continue
            for value in values:
                if key == 'preds' or key == 'labels':
                    if i == 0:
                        all_dicts[key] = value
                    else:
                        all_dicts[key] = np.concatenate([all_dicts[key], value])
                else:
                    if i == 0:
                        all_dicts[key] = [value,]
                    else:
                        all_dicts[key].append(value)
        i = i + 1

    return all_dicts

pkl_list_file1 = sys.argv[1]
outdir = sys.argv[2]
tmpdir = f"{outdir}/ALL/"

try:
    pkl_list_file2 = sys.argv[3]
except:
    pass

all_dicts1 = merge_pkl_files(pkl_list_file1)
all_preds1 = all_dicts1["preds"].flatten()
print(all_preds1.shape)
all_preds1 = np.expand_dims(all_preds1, axis=0)
all_labels1 = all_dicts1["labels"].flatten()
all_labels1 = np.expand_dims(all_labels1, axis=0)
#datetimes = all_dicts["datetimes"]
#stride = all_dicts["stride"]

#try:
all_dicts2 = merge_pkl_files(pkl_list_file2)
all_preds2 = all_dicts2["preds"].flatten()
all_labels2 = all_dicts2["labels"].flatten()
all_preds2 = np.expand_dims(all_preds2, axis=0)
all_labels2 = np.expand_dims(all_labels2, axis=0)

all_preds = np.concatenate([all_preds1, all_preds2], axis=0)
print(all_preds.shape)
all_labels = np.concatenate([all_labels1, all_labels2])
scores_dict = verification(all_preds, all_labels, tmpdir)
#except:
#    pass


