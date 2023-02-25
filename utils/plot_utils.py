import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame


def plot_summary(summary_df: DataFrame, title: str):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    n_epochs = max(list(summary_df.step))
    sns.set_theme(palette="colorblind", style="whitegrid")
    plot = sns.lineplot(x="step", y="value", hue="model", data=summary_df)
    plot.set(title=title, xlabel="Epoch", ylabel="Accuracy")
    plot.set_xlim(0, n_epochs)
    plt.legend(loc="lower right")


def plot_gradient_distribution_comparison(model1_grad_df: DataFrame, model1_title: str, model2_grad_df: DataFrame,
                                          model2_title: str):
    n_layers = int(model1_grad_df.iloc[-1].tag[24])
    fig, axs = plt.subplots(n_layers + 1, 2, sharex="col", sharey="row", figsize=(10, 12),
                            gridspec_kw={"height_ratios": [0.02, 1, 1, 1, 1]})

    sns.set_theme(palette="Blues", style="whitegrid")

    for i in range(1, n_layers + 1):
        current_layer = f"Gradient of dense layer {i} weights"

        model1_layer_df = model1_grad_df[model1_grad_df.tag == current_layer]
        plot1 = sns.boxplot(x="step", y="limits", data=model1_layer_df, showfliers=False, ax=axs[i, 0])
        plot1.set(title=f"Layer {i}", xlabel="", ylabel="")

        model2_layer_df = model2_grad_df[model2_grad_df.tag == current_layer]
        plot2 = sns.boxplot(x="step", y="limits", data=model2_layer_df, showfliers=False, ax=axs[i, 1])
        plot2.set(title=f"Layer {i}", xlabel="", ylabel="")

    axs[0, 0].axis("off")
    axs[0, 0].set_title(model1_title)

    axs[0, 1].axis("off")
    axs[0, 1].set_title(model2_title)

    fig.suptitle("Weights Gradient Distribution Comparison")
    fig.supxlabel("Update step")
    fig.supylabel("Gradient value")
    fig.subplots_adjust(hspace=0.5, bottom=0.1)
    plt.show()
