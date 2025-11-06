"""
Visualization utilities for the Titanic Survival Prediction project.

This module contains functions for creating consistent, publication-quality
visualizations throughout the project.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from pathlib import Path


def set_plot_style(style: str = 'seaborn-v0_8-whitegrid', palette: str = 'husl') -> None:
    """
    Set consistent plotting style for all visualizations.
    
    Args:
        style: Matplotlib style to use
        palette: Seaborn color palette
    """
    plt.style.use(style)
    sns.set_palette(palette)
    sns.set_context("notebook", font_scale=1.1)


def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None
) -> matplotlib.figure.Figure:
    """
    Plot confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_xticklabels(['Did Not Survive', 'Survived'])
    ax.set_yticklabels(['Did Not Survive', 'Survived'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_proba: Union[np.ndarray, pd.Series],
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None
) -> matplotlib.figure.Figure:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results_dict: Dict[str, float],
    title: str = 'Model Performance Comparison',
    metric_name: str = 'Accuracy',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> matplotlib.figure.Figure:
    """
    Create horizontal bar chart comparing model performances.
    
    Args:
        results_dict: Dictionary mapping model names to scores
        title: Plot title
        metric_name: Name of the metric being compared
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    # Sort by performance
    sorted_results = dict(sorted(results_dict.items(), key=lambda item: item[1]))
    
    models = list(sorted_results.keys())
    scores = list(sorted_results.values())
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(models, scores, color=sns.color_palette('husl', len(models)))
    
    # Add value labels
    for i, (_, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.005, i, f'{score:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(scores) * 1.1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = 'Feature Importance',
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> matplotlib.figure.Figure:
    """
    Plot feature importances as horizontal bar chart.
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importance values
        title: Plot title
        top_n: Number of top features to display
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    # Create dataframe and sort
    feat_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    _ = ax.barh(
        feat_importance_df['feature'], 
        feat_importance_df['importance'],
        color=sns.color_palette('viridis', len(feat_importance_df))
    )
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_learning_curves(
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: List[int],
    title: str = 'Learning Curves',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> matplotlib.figure.Figure:
    """
    Plot learning curves showing training and validation scores.
    
    Args:
        train_scores: Training scores
        val_scores: Validation scores
        train_sizes: Training set sizes
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_sizes, train_scores, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, val_scores, 'o-', color='g', label='Validation score')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    title: str = 'Feature Correlation Heatmap',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None
) -> matplotlib.figure.Figure:
    """
    Plot correlation heatmap for features.
    
    Args:
        data: DataFrame containing features
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    correlation = data.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        correlation,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cv_score_distribution(
    cv_scores: Dict[str, List[float]],
    title: str = 'Cross-Validation Score Distribution',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> matplotlib.figure.Figure:
    """
    Plot distribution of cross-validation scores for multiple models.
    
    Args:
        cv_scores: Dictionary mapping model names to lists of CV scores
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for boxplot
    data = []
    labels = []
    for model_name, scores in cv_scores.items():
        data.append(scores)
        labels.append(model_name)
    
    positions = list(range(1, len(data) + 1))
    bp = ax.boxplot(data, positions=positions, patch_artist=True, showmeans=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    
    # Customize boxplot colors
    colors = sns.color_palette('husl', len(data))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_proba: Union[np.ndarray, pd.Series],
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None
) -> matplotlib.figure.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color='darkorange', lw=2, label='PR curve')
    ax.axhline(y=y_true.mean(), color='navy', linestyle='--', label='Baseline (random)')
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
