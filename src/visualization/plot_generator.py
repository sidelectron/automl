"""Plot generator for EDA visualizations with multiple backend support."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

# Try to import plotting libraries in order of preference
PLOTLY_AVAILABLE = False
SEABORN_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    PLOTTING_BACKEND = "plotly"
except ImportError:
    go = None
    px = None
    make_subplots = None

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    SEABORN_AVAILABLE = True
    if not PLOTLY_AVAILABLE:
        PLOTTING_BACKEND = "seaborn"
except ImportError:
    sns = None
    if not PLOTLY_AVAILABLE:
        plt = None

try:
    if not SEABORN_AVAILABLE:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
        if not PLOTLY_AVAILABLE:
            PLOTTING_BACKEND = "matplotlib"
except ImportError:
    if not SEABORN_AVAILABLE:
        plt = None

# Check if any plotting library is available
if not (PLOTLY_AVAILABLE or SEABORN_AVAILABLE or MATPLOTLIB_AVAILABLE):
    raise ImportError(
        "No plotting library available. Install one of: plotly, seaborn, or matplotlib"
    )


class PlotGenerator:
    """Generate visualizations for EDA using available plotting library."""

    def __init__(self):
        """Initialize plot generator."""
        self.backend = PLOTTING_BACKEND
        if self.backend == "plotly":
            if not PLOTLY_AVAILABLE:
                raise ImportError("plotly not available")
        elif self.backend == "seaborn":
            if not SEABORN_AVAILABLE:
                raise ImportError("seaborn not available")
        elif self.backend == "matplotlib":
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("matplotlib not available")

    def generate_plot(
        self,
        df: pd.DataFrame,
        visualization_spec: Dict[str, Any],
        target_variable: Optional[str] = None
    ) -> Any:
        """Generate a plot from visualization specification.

        Args:
            df: DataFrame with data
            visualization_spec: Visualization specification dictionary
            target_variable: Optional target variable name

        Returns:
            Figure object (plotly Figure, matplotlib Figure, or None for seaborn)
        """
        plot_type = visualization_spec.get("type", "histogram")
        title = visualization_spec.get("title", "Plot")
        columns = visualization_spec.get("columns", [])

        if plot_type == "histogram":
            return self._create_histogram(df, columns[0] if columns else None, title)

        elif plot_type == "bar":
            return self._create_bar_chart(df, columns, target_variable, title)

        elif plot_type == "scatter":
            return self._create_scatter(df, columns, target_variable, title)

        elif plot_type == "correlation":
            return self._create_correlation_heatmap(df, target_variable, title)

        elif plot_type == "boxplot":
            return self._create_boxplot(df, columns[0] if columns else None, target_variable, title)

        elif plot_type == "kde" or plot_type == "distribution":
            return self._create_kde(df, columns[0] if columns else None, target_variable, title)

        elif plot_type == "pairplot":
            return self._create_pairplot(df, columns, target_variable, title)

        elif plot_type == "qq" or plot_type == "qq_plot":
            return self._create_qq_plot(df, columns[0] if columns else None, title)

        elif plot_type == "violin":
            return self._create_violin(df, columns[0] if columns else None, target_variable, title)

        else:
            # Default: simple histogram
            return self._create_histogram(df, columns[0] if columns else df.columns[0], title)

    def _create_histogram(
        self,
        df: pd.DataFrame,
        column: Optional[str],
        title: str
    ) -> Any:
        """Create histogram plot."""
        if column is None:
            column = df.columns[0]

        if self.backend == "plotly":
            return px.histogram(df, x=column, title=title)
        elif self.backend == "seaborn":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=column, ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            return fig
        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[column].dropna(), bins=30)
            ax.set_title(title)
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            return fig

    def _create_bar_chart(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_variable: Optional[str],
        title: str
    ) -> Any:
        """Create bar chart (target-focused)."""
        if not columns:
            columns = [df.columns[0]]

        if target_variable and target_variable in df.columns and columns[0] in df.columns:
            # Calculate target rate by categorical feature
            target_rate = df.groupby(columns[0])[target_variable].mean()
            
            if self.backend == "plotly":
                return px.bar(
                    x=target_rate.index,
                    y=target_rate.values,
                    title=title,
                    labels={"x": columns[0], "y": f"{target_variable} Rate"}
                )
            elif self.backend == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=target_rate.index, y=target_rate.values, ax=ax)
                ax.set_title(title)
                ax.set_xlabel(columns[0])
                ax.set_ylabel(f"{target_variable} Rate")
                plt.tight_layout()
                return fig
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(target_rate.index, target_rate.values)
                ax.set_title(title)
                ax.set_xlabel(columns[0])
                ax.set_ylabel(f"{target_variable} Rate")
                plt.tight_layout()
                return fig
        else:
            # Simple bar chart
            if self.backend == "plotly":
                return px.bar(df, x=columns[0], title=title)
            elif self.backend == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(data=df, x=columns[0], ax=ax)
                ax.set_title(title)
                plt.tight_layout()
                return fig
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                df[columns[0]].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(title)
                ax.set_xlabel(columns[0])
                plt.tight_layout()
                return fig

    def _create_scatter(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_variable: Optional[str],
        title: str
    ) -> Any:
        """Create scatter plot."""
        if len(columns) >= 1 and target_variable and target_variable in df.columns:
            x_col = columns[0]
            if x_col in df.columns:
                if self.backend == "plotly":
                    return px.scatter(df, x=x_col, y=target_variable, title=title)
                elif self.backend == "seaborn":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=df, x=x_col, y=target_variable, ax=ax)
                    ax.set_title(title)
                    plt.tight_layout()
                    return fig
                else:  # matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df[x_col], df[target_variable])
                    ax.set_title(title)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(target_variable)
                    plt.tight_layout()
                    return fig

        # Fallback: use first two columns
        if len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
        else:
            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        if self.backend == "plotly":
            return px.scatter(df, x=x_col, y=y_col, title=title)
        elif self.backend == "seaborn":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            return fig
        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[x_col], df[y_col])
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.tight_layout()
            return fig

    def _create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str],
        title: str
    ) -> Any:
        """Create correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            # Fallback to simple plot
            return self._create_histogram(df, df.columns[0], title)

        corr = numeric_df.corr()

        if target_variable and target_variable in corr.columns:
            # Focus on correlations with target
            target_corr = corr[target_variable].sort_values(ascending=False)
            
            if self.backend == "plotly":
                return px.bar(
                    x=target_corr.index,
                    y=target_corr.values,
                    title=f"{title} - Correlation with {target_variable}",
                    labels={"x": "Feature", "y": "Correlation"}
                )
            elif self.backend == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=target_corr.index, y=target_corr.values, ax=ax)
                ax.set_title(f"{title} - Correlation with {target_variable}")
                ax.set_xlabel("Feature")
                ax.set_ylabel("Correlation")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                return fig
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(target_corr.index, target_corr.values)
                ax.set_title(f"{title} - Correlation with {target_variable}")
                ax.set_xlabel("Feature")
                ax.set_ylabel("Correlation")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                return fig
        else:
            # Full correlation matrix
            if self.backend == "plotly":
                return px.imshow(
                    corr,
                    title=title,
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
            elif self.backend == "seaborn":
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
                ax.set_title(title)
                plt.tight_layout()
                return fig
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr, cmap='RdBu_r', aspect='auto')
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha='right')
                ax.set_yticklabels(corr.columns)
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                return fig

    def _create_boxplot(
        self,
        df: pd.DataFrame,
        column: Optional[str],
        target_variable: Optional[str],
        title: str
    ) -> Any:
        """Create boxplot for outlier visualization.

        From ML text Pages 127-134: Boxplots show IQR, median, and outliers.

        Args:
            df: DataFrame with data
            column: Column to plot
            target_variable: Optional grouping variable
            title: Plot title

        Returns:
            Figure object
        """
        if column is None:
            # Find first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        if self.backend == "plotly":
            if target_variable and target_variable in df.columns:
                return px.box(df, x=target_variable, y=column, title=title)
            else:
                return px.box(df, y=column, title=title)

        elif self.backend == "seaborn":
            fig, ax = plt.subplots(figsize=(10, 6))
            if target_variable and target_variable in df.columns:
                sns.boxplot(data=df, x=target_variable, y=column, ax=ax)
            else:
                sns.boxplot(data=df, y=column, ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            return fig

        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            if target_variable and target_variable in df.columns:
                groups = df.groupby(target_variable)[column].apply(list)
                ax.boxplot([g for g in groups.values], labels=groups.index)
                ax.set_xlabel(target_variable)
            else:
                ax.boxplot(df[column].dropna())
            ax.set_ylabel(column)
            ax.set_title(title)
            plt.tight_layout()
            return fig

    def _create_kde(
        self,
        df: pd.DataFrame,
        column: Optional[str],
        target_variable: Optional[str],
        title: str
    ) -> Any:
        """Create KDE/distribution plot.

        Shows probability density function of the data distribution.

        Args:
            df: DataFrame with data
            column: Column to plot
            target_variable: Optional grouping variable for comparison
            title: Plot title

        Returns:
            Figure object
        """
        if column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        if self.backend == "plotly":
            if target_variable and target_variable in df.columns:
                # KDE by group
                fig = go.Figure()
                for group_val in df[target_variable].unique():
                    group_data = df[df[target_variable] == group_val][column].dropna()
                    if len(group_data) > 1:
                        fig.add_trace(go.Histogram(
                            x=group_data,
                            name=str(group_val),
                            histnorm='probability density',
                            opacity=0.7
                        ))
                fig.update_layout(title=title, barmode='overlay')
                return fig
            else:
                return px.histogram(
                    df, x=column, title=title,
                    histnorm='probability density',
                    marginal='box'
                )

        elif self.backend == "seaborn":
            fig, ax = plt.subplots(figsize=(10, 6))
            if target_variable and target_variable in df.columns:
                for group_val in df[target_variable].unique():
                    group_data = df[df[target_variable] == group_val][column].dropna()
                    if len(group_data) > 1:
                        sns.kdeplot(data=group_data, ax=ax, label=str(group_val))
                ax.legend(title=target_variable)
            else:
                sns.kdeplot(data=df, x=column, ax=ax, fill=True)
            ax.set_title(title)
            plt.tight_layout()
            return fig

        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            data = df[column].dropna()
            # Simple histogram approximation of KDE
            ax.hist(data, bins=50, density=True, alpha=0.7)
            ax.set_xlabel(column)
            ax.set_ylabel('Density')
            ax.set_title(title)
            plt.tight_layout()
            return fig

    def _create_pairplot(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_variable: Optional[str],
        title: str
    ) -> Any:
        """Create pairplot for multivariate analysis.

        Shows relationships between multiple numeric variables.

        Args:
            df: DataFrame with data
            columns: Columns to include (max 5 for readability)
            target_variable: Optional color grouping
            title: Plot title

        Returns:
            Figure object
        """
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if columns:
            # Filter to requested columns that exist and are numeric
            valid_cols = [c for c in columns if c in numeric_df.columns]
            if valid_cols:
                numeric_df = numeric_df[valid_cols]

        # Limit to max 5 columns for readability
        if len(numeric_df.columns) > 5:
            numeric_df = numeric_df[numeric_df.columns[:5]]

        if len(numeric_df.columns) < 2:
            # Fall back to histogram if not enough columns
            return self._create_histogram(df, numeric_df.columns[0] if len(numeric_df.columns) > 0 else df.columns[0], title)

        if self.backend == "plotly":
            if target_variable and target_variable in df.columns:
                return px.scatter_matrix(
                    df,
                    dimensions=numeric_df.columns.tolist(),
                    color=target_variable,
                    title=title
                )
            else:
                return px.scatter_matrix(
                    df,
                    dimensions=numeric_df.columns.tolist(),
                    title=title
                )

        elif self.backend == "seaborn":
            if target_variable and target_variable in df.columns:
                plot_df = pd.concat([numeric_df, df[[target_variable]]], axis=1)
                g = sns.pairplot(plot_df, hue=target_variable, diag_kind='kde')
            else:
                g = sns.pairplot(numeric_df, diag_kind='kde')
            g.figure.suptitle(title, y=1.02)
            return g.figure

        else:  # matplotlib
            n_cols = len(numeric_df.columns)
            fig, axes = plt.subplots(n_cols, n_cols, figsize=(12, 12))

            for i, col1 in enumerate(numeric_df.columns):
                for j, col2 in enumerate(numeric_df.columns):
                    ax = axes[i, j] if n_cols > 1 else axes
                    if i == j:
                        # Diagonal: histogram
                        ax.hist(numeric_df[col1].dropna(), bins=20, alpha=0.7)
                    else:
                        # Off-diagonal: scatter
                        ax.scatter(numeric_df[col2], numeric_df[col1], alpha=0.5, s=10)

                    if i == n_cols - 1:
                        ax.set_xlabel(col2)
                    if j == 0:
                        ax.set_ylabel(col1)

            fig.suptitle(title)
            plt.tight_layout()
            return fig

    def _create_qq_plot(
        self,
        df: pd.DataFrame,
        column: Optional[str],
        title: str
    ) -> Any:
        """Create Q-Q plot for normality check.

        From ML text: Q-Q plots compare data quantiles against
        theoretical normal distribution quantiles.

        Args:
            df: DataFrame with data
            column: Column to check for normality
            title: Plot title

        Returns:
            Figure object
        """
        if column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        data = df[column].dropna().values
        data_sorted = np.sort(data)
        n = len(data_sorted)

        # Calculate theoretical quantiles
        theoretical_quantiles = np.array([
            (i - 0.5) / n for i in range(1, n + 1)
        ])

        # Convert to standard normal quantiles
        from scipy import stats
        theoretical_values = stats.norm.ppf(theoretical_quantiles)

        if self.backend == "plotly":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=theoretical_values,
                y=data_sorted,
                mode='markers',
                name='Data'
            ))
            # Add reference line
            min_val = min(theoretical_values.min(), data_sorted.min())
            max_val = max(theoretical_values.max(), data_sorted.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Reference',
                line=dict(dash='dash', color='red')
            ))
            fig.update_layout(
                title=f"{title} - Q-Q Plot for {column}",
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles'
            )
            return fig

        elif self.backend == "seaborn":
            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"{title} - Q-Q Plot for {column}")
            plt.tight_layout()
            return fig

        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(theoretical_values, data_sorted, alpha=0.7)
            # Reference line
            min_val = min(theoretical_values.min(), data_sorted.min())
            max_val = max(theoretical_values.max(), data_sorted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Reference')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.set_title(f"{title} - Q-Q Plot for {column}")
            ax.legend()
            plt.tight_layout()
            return fig

    def _create_violin(
        self,
        df: pd.DataFrame,
        column: Optional[str],
        target_variable: Optional[str],
        title: str
    ) -> Any:
        """Create violin plot combining boxplot and KDE.

        Shows both distribution shape and summary statistics.

        Args:
            df: DataFrame with data
            column: Column to plot
            target_variable: Optional grouping variable
            title: Plot title

        Returns:
            Figure object
        """
        if column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        if self.backend == "plotly":
            if target_variable and target_variable in df.columns:
                return px.violin(
                    df, x=target_variable, y=column,
                    box=True, points='outliers',
                    title=title
                )
            else:
                return px.violin(df, y=column, box=True, points='outliers', title=title)

        elif self.backend == "seaborn":
            fig, ax = plt.subplots(figsize=(10, 6))
            if target_variable and target_variable in df.columns:
                sns.violinplot(data=df, x=target_variable, y=column, ax=ax)
            else:
                sns.violinplot(data=df, y=column, ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            return fig

        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            if target_variable and target_variable in df.columns:
                groups = df.groupby(target_variable)[column].apply(list)
                ax.violinplot([g for g in groups.values], positions=range(len(groups)))
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels(groups.index)
                ax.set_xlabel(target_variable)
            else:
                ax.violinplot(df[column].dropna())
            ax.set_ylabel(column)
            ax.set_title(title)
            plt.tight_layout()
            return fig
