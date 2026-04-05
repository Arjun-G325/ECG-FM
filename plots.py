from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import os


mpl.rcParams["font.family"]='Liberation Serif'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 8

 
 ##############################################################################
def plot_loss_and_mae(history, title_lloss, title_mae, save_path=None, dpi=300):
    """
    Plot the loss and MAE for both training and validation, and save the figure.
    
    Parameters:
    - history: Keras History object
    - title_lloss: Title for loss subplot
    - title_mae: Title for MAE subplot
    - save_path: Path to save the figure (including filename)
    - dpi: Dots per inch (for high-resolution images)
    """
    
  

    plt.figure(figsize=(12, 15)) 

    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title_lloss, )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

  
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(title_mae, )
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()



def ecg_plot_test(y_test, y_pred, title, save_path=None, dpi=300):
    """
    Scatter plot of Actual vs Predicted values with a dotted line for perfect predictions,
    a regression line for fitted predictions, and R² value in the legend.
    
    Parameters:
    - y_test: True labels (actual values).
    - y_pred: Predicted labels from the model.
    - title: Title of the plot.
    - save_path: Optional path to save the figure (PNG).
    - dpi: Resolution for saved figure (default: 300).
    """
    
    if len(y_test) != len(y_pred):
        raise ValueError("The length of y_test and y_pred must be the same.")

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Fit regression model
    model = LinearRegression()
    model.fit(y_test.reshape(-1, 1), y_pred)
    y_fit_line = model.predict(y_test.reshape(-1, 1))

    # Compute R²
    r2 = r2_score(y_test, y_pred)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction (y = x)')
    plt.plot(y_test, y_fit_line, color='green', linestyle='-', linewidth=2, label=f'Fitted Line (R² = {r2:.3f})')
    
    plt.title(title)
    plt.xlabel('Chronological Age')
    plt.ylabel('Biological Age')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    



def plot_combined_histogram(y_true, y_pred, save_path=None,bins=20, kde=True, alpha=0.5, color_true="blue", color_pred="green", 
                            xlabel="Value", ylabel="Frequency", title="True vs Predicted Distribution",  dpi=300):
    """
    Plot histogram of true and predicted values on the same plot.

    Parameters:
    - y_true, y_pred: Ground truth and predicted values.
    - bins: Number of bins.
    - kde: Whether to show KDE.
    - alpha: Transparency of histograms.
    - color_true, color_pred: Colors for true and predicted.
    - xlabel, ylabel, title: Labels and title.
    - save_path: If provided, saves the plot to this path.
    - dpi: Resolution for saved image.
    """
    plt.figure(figsize=(20, 10))
    
    sns.histplot(y_true, bins=bins, kde=kde, color=color_true, label="True", alpha=alpha, stat="density")
    sns.histplot(y_pred, bins=bins, kde=kde, color=color_pred, label="Predicted", alpha=alpha, stat="density")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()




def plot_boxplot_bins(y_actual, y_pred,save_path=None, bin_size=10, title="Boxplot of Predictions by Actual Value Bins",
                       dpi=300):
    """
    Plots a boxplot comparing actual and predicted values, grouped into bins.

    Parameters:
    - y_actual: Actual (true) values.
    - y_pred: Predicted values.
    - bin_size: Width of each bin (default=10).
    - title: Plot title.
    - save_path: If provided, saves the figure to this path.
    - dpi: Image resolution.
    """
    if len(y_actual) != len(y_pred):
        raise ValueError("y_actual and y_pred must have the same length.")

   
    bins = np.arange(0, 110, bin_size)
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

  
    df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})
    df['Bin'] = pd.cut(df['Actual'], bins=bins, labels=bin_labels, include_lowest=True)

    df_melted = df.melt(id_vars=['Bin'], value_vars=['Actual', 'Predicted'],
                        var_name='Type', value_name='Value')

    # Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Bin', y='Value', hue='Type', data=df_melted, palette='Set2')

    plt.xlabel("Bins (Range of Actual Values)")
    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend(title="Data Type")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_mae_by_age_bin(y_pred, y_test,save_path, bin_size=10, age_max=100,age_min=0, title="Bin-wise MAE by Age Group",
                         dpi=300):
    """
    Computes MAE between y_pred and y_test, then plots the mean MAE grouped by age bins.

    Parameters:
    - y_pred: Predicted values (array-like)
    - y_test: True values (array-like)
    - bin_size: Width of age bins (default=10)
    - age_max: Maximum age to consider in bins (default=100)
    - title: Title of the plot
    - save_path: If provided, saves the plot to this path.
    - dpi: Image resolution.
    """
    
    y_pred = np.array(y_pred).flatten()
    y_test = np.array(y_test)

    mae_list = np.abs(y_test - y_pred)
    df = pd.DataFrame({
        "y_true": y_test,
        "MAE": mae_list
    })

    df["MAE"] = pd.to_numeric(df["MAE"], errors='coerce')
    df.dropna(subset=["y_true", "MAE"], inplace=True)

  
    bin_edges = np.arange(age_min, age_max + bin_size, bin_size)
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]
    df["age_bin"] = pd.cut(df["y_true"], bins=bin_edges, labels=bin_labels, right=False)


    grouped = df.groupby("age_bin").agg(
        mean_mae=("MAE", "mean"),
        count=("MAE", "count")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(grouped["age_bin"], grouped["mean_mae"], color="skyblue", edgecolor="black")

    for bar, count in zip(bars, grouped["count"]):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, str(count),
                    ha='center', va='bottom', fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Age Group (Years)")
    ax.set_ylabel("Mean MAE")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

    ###############################################
def ensure_unique_folder_name(model_name, dataset_name):
    """
    Ensure the folder name is unique by appending a timestamp or version number
    if a folder with the same name already exists.
    """
    base_folder_name = f"./{model_name}_{dataset_name}"
    folder_name = base_folder_name
    version = 1
    while os.path.exists(folder_name):
        folder_name = f"{base_folder_name}_v{version}"
        version += 1
        
    return folder_name


