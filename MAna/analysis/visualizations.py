def set_matplotlib_defaults(labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10, cmap='magma'):
    import matplotlib.pyplot as plt
    import warnings

    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelweight=labelweight, labelsize=labelsize,
        titleweight=titleweight, titlesize=titlesize, titlepad=titlepad)
    plt.rc('image', cmap=cmap)
    warnings.filterwarnings("ignore") # to clean up output cells


def line(df, x_col, y_col, title='', x_title='', y_title='', color='blue', width=None, height=None, show_legend=True, filename=None):
    """
    Creates a line plot from a pandas DataFrame using Plotly.

    Parameters:
    df (pandas DataFrame): The DataFrame to plot.
    x_col (str): The name of the column to use for the x-axis.
    y_col (str): The name of the column to use for the y-axis.
    title (str): The title of the plot.
    x_title (str): The title of the x-axis.
    y_title (str): The title of the y-axis.
    color (str): The color of the line.
    width (int): The width of the plot in pixels.
    height (int): The height of the plot in pixels.
    show_legend (bool): Whether to show the legend or not.
    filename (str): The name of the file to save the plot to (including file extension). If not provided, the plot will not be saved.

    Returns:
    None
    """
    import plotly.graph_objs as go
    import os

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='lines', line_color=color, name=y_col))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, width=width, height=height, showlegend=show_legend)
    fig.show()

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath)


def scatter(df, x_col, y_col, color_col=None, size_col=None, title=None, x_title=None, y_title=None, width=None, height=None, template='plotly_white', mode='markers', symbol='circle', opacity=0.7, marker=None, filename=None, auto_open=True) -> object:
    """
    Outputs a scatter plot based on data from a data frame using plotly.

    Parameters:
        df (pandas.DataFrame): The data frame containing the data to plot.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        color_col (str, optional): The name of the column to use for coloring the data points.
        size_col (str, optional): The name of the column to use for sizing the data points.
        title (str, optional): The title of the plot.
        x_title (str, optional): The title of the x-axis.
        y_title (str, optional): The title of the y-axis.
        width (int, optional): The width of the plot in pixels.
        height (int, optional): The height of the plot in pixels.
        template (str, optional): The plotly template to use.
        mode (str, optional): The mode of the plot (markers or lines).
        symbol (str, optional): The symbol of the markers (only used if mode='markers').
        opacity (float, optional): The opacity of the markers (only used if mode='markers').
        marker (dict, optional): A dictionary of marker options (only used if mode='markers').
        filename (str, optional): The filename to use for saving the plot. If not provided, the plot will not be saved.
        auto_open (bool, optional): Whether to automatically open the plot in a new browser tab.

    Returns:
        object: The Plotly figure object.
    """
    import os
    import plotly.express as px

    labels = {}
    if x_title is not None:
        labels[x_col] = x_title
    if y_title is not None:
        labels[y_col] = y_title

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        labels=labels,
        width=width,
        height=height,
        template=template,
        opacity=opacity,
        trendline="ols",
        trendline_color_override="red",
    )

    fig.update_traces(mode=mode, marker_symbol=symbol)
    if marker:
        fig.update_traces(marker=marker)

    if filename:
        if not filename.endswith('.html'):
            filename = filename + '.html'

        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    return fig


def bar(df, x_col, y_col, color_col=None, title='', x_title='', y_title='', width=None, height=None, template='plotly_white', filename=None, auto_open=True):
    """
    Creates a bar chart from a pandas DataFrame using Plotly.

    Parameters:
    df (pandas DataFrame): The DataFrame to plot.
    x_col (str): The name of the column to use for the x-axis.
    y_col (str): The name of the column to use for the y-axis.
    color_col (str, optional): The name of the column to use for coloring the bars.
    title (str): The title of the plot.
    x_title (str): The title of the x-axis.
    y_title (str): The title of the y-axis.
    width (int): The width of the plot in pixels.
    height (int): The height of the plot in pixels.
    template (str): The plotly template to use.
    filename (str, optional): The filename to use for saving the plot. If not provided, the plot will not be saved.
    auto_open (bool): Whether to automatically open the plot in a new browser tab.

    Returns:
    None
    """
    import plotly.express as px
    import os

    fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title, labels={x_col: x_title, y_col: y_title}, width=width, height=height, template=template)

    if filename:
        if not filename.endswith('.html'):
            filename = filename + '.html'

        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    fig.show()


def dist(df,col,lenght = 4, filename=None, auto_open = False, **kwargs):
    """
    Function to plot the distribution of a column in a given pandas DataFrame.

    Args:
    df (pandas DataFrame): The DataFrame to use for plotting.
    col (str): The name of the column to plot the distribution for.
    lenght (float, optional): The length of the vertical axis in multiples of the DataFrame length. Defaults to 4.
    filename (str, optional): The filename to use for saving the plot. If not provided, the plot will not be saved.
    auto_open (bool): Whether to automatically open the plot in a new browser tab.

    Returns:
    None
    """

    """
    The function first calculates the length of the vertical axis for the plot based on the length of the DataFrame and the
    specified length parameter.

    Then, it creates a subplot with a single plot area and adds a histogram trace with the data from the specified column.
    Next, it adds three vertical line traces to the plot, one at the mean, one at the median, and one at the mode of the
    specified column. The lines are placed using the length calculated previously, and labeled with the respective statistics.

    Finally, the function shows the plot using the plotly show() method.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go
    import os


    vertical_length = len(df)*lenght/10 # count some lenght first,

    fig = make_subplots(rows=1, cols=1) # then creates a subplot and..

    fig.add_trace(go.Histogram(x=df[col], name=col)) # give it a histogram plot that shows the distripution,
    fig.add_trace(go.Scatter(x=[df[col].mean() for i in range(round(vertical_length))], y=list(range(round(vertical_length))), mode='lines', name=f'{col}\'s mean')) # a virtical line that shows where the mean is,
    fig.add_trace(go.Scatter(x=[df[col].median() for i in range(round(vertical_length))], y=list(range(round(vertical_length))), mode='lines', name=f'{col}\'s median')) # a virtical line that shows where the median is and
    fig.add_trace(go.Scatter(x=[df[col].mode()[0] for i in range(round(vertical_length))], y=list(range(round(vertical_length))), mode='lines', name=f'{col}\'s mode')) # a virtical line that shows where the mode is.
    fig.update_layout(**kwargs)

    if filename:
        if not filename.endswith('.html'):
            filename = filename + '.html'

        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    fig.show()


def plot_box(df, x_col, y_col, title='', x_title='', y_title='', color='blue', width=None, height=None, show_legend=True):
    """
    Creates a box plot from a pandas DataFrame using Plotly.

    Parameters:
    df (pandas DataFrame): The DataFrame to plot.
    x_col (str): The name of the column to use for the x-axis.
    y_col (str): The name of the column to use for the y-axis.
    title (str): The title of the plot.
    x_title (str): The title of the x-axis.
    y_title (str): The title of the y-axis.
    color (str): The color of the box.
    width (int): The width of the plot in pixels.
    height (int): The height of the plot in pixels.
    show_legend (bool): Whether to show the legend or not.
    download_path (str): The path to save the downloaded plot. Default is the current working directory.

    Returns:
    None
    """
    import plotly.graph_objs as go

    fig = go.Figure()
    fig.add_trace(go.Box(x=df[x_col], y=df[y_col], marker_color=color, name=y_col))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, width=width, height=height, showlegend=show_legend)
    fig.show()

    return fig


def heatmap(df, figsize=(15, 15), cmap = "Greens",linewidths=0.1, annot_kws={"fontsize":10}):
    """
    Generates a heatmap based on the correlation matrix of the provided DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame to generate the heatmap from.

    Returns:
    --------
    seaborn matrix plot
        The resulting heatmap plot showing the correlation matrix between the columns.

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the size of the heatmap
    plt.figure(figsize=figsize)

    # Generate the heatmap with seaborn
    return sns.heatmap(df.corr(), annot=True, cmap=cmap, linewidths=linewidths, annot_kws=annot_kws)


def pairplot(df, color=None, size=None, show=True):
    """
    A function to plot a pair plot for a given dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to plot the pair plot for.
    color : str, array-like, or None, optional (default=None)
        A column name or array-like values to use for color encoding the scatter plot matrix.
    size : str, number, or None, optional (default=None)
        A column name of df to use for size encoding the scatter plot matrix,
        or a fixed marker size such as 20.
    show : bool, optional (default=True)
        Whether to display the figure immediately.

    Returns:
    --------
    plotly.graph_objs.Figure
        The scatter matrix figure.

    Example:
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    >>> df["species"] = iris.target
    >>> pairplot(df, color="species", size=20)
    """

    import plotly.express as px

    fig = px.scatter_matrix(df, color=color)
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(width=size, height=size)
    if show:
        fig.show()
    return fig

def pairplot_seaborn(df, hue=None, kind = 'scatter', diag_kind='hist', palette=None, **kwargs):
    """
    A function to plot a pair plot for a given dataframe using seaborn.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to plot the pair plot for.
    hue : str, optional (default=None)
        A column name of df to use for color encoding the scatter plot matrix.
    diag_kind : str, optional (default='hist')
        The type of plot to use on the diagonal. Options are 'hist' or 'kde'.
    palette : dict or seaborn color palette, optional (default=None)
        A dictionary mapping hue levels to colors, or a seaborn color palette.
    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to seaborn.pairplot().

    Returns:
    --------
    seaborn.axisgrid.PairGrid
        The pair plot object.

    Example:
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    >>> df["species"] = iris.target
    >>> pairplot_seaborn(df, hue="species", diag_kind='kde', palette='Set2')
    """
    import seaborn as sns

    return sns.pairplot(df, hue=hue, kind = kind, diag_kind=diag_kind, palette=palette, **kwargs)


def area_plot(df, x_col, y_col, title=None, x_title=None, y_title=None, color=None, filename=None):
    """
    A function to create an Area plot based on data from a DataFrame using Plotly.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    x_col (str): The name of the column in the DataFrame to use as the x-axis data.
    y_col (str): The name of the column in the DataFrame to use as the y-axis data.
    title (str): The title of the plot.
    x_title (str): The title of the x-axis.
    y_title (str): The title of the y-axis.
    color (str): The name of the column in the DataFrame to use as the color data.
    filename (str): The filename to save the plot as. If not specified, the plot will not be saved.

    Returns:
    None
    """
    import plotly.express as px

    fig = px.area(df, x=x_col, y=y_col, color=color, title=title)
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    if filename:
        fig.write_html(filename)
    fig.show()


def sunburst(df, hierarchy_cols, size_col, color_col=None, title=None, width=800, height=800, font_size=14, colorscale='YlOrRd', download_path=None, show=True):
    """
    Create a sunburst graph using Plotly based on data from a data frame.

    Args:
    - df: pandas DataFrame containing the data for the sunburst graph.
    - hierarchy_cols: list of column names to use for the hierarchy of the sunburst graph.
    - size_col: name of the column to use for the size of the segments in the sunburst graph.
    - color_col: name of the column to use for the color of the segments in the sunburst graph.
    - title: title of the sunburst graph.
    - width: width of the sunburst graph in pixels.
    - height: height of the sunburst graph in pixels.
    - font_size: font size for the text in the sunburst graph.
    - colorscale: name of the Plotly colorscale to use for the color of the segments.
    - download_path: file path for downloading the sunburst graph as an HTML file. If None, the graph will not be downloaded.

    Returns:
    - fig: Plotly figure object for the sunburst graph.
    """
    import plotly.express as px

    fig = px.sunburst(
        df,
        path=hierarchy_cols,
        values=size_col,
        color=color_col,
        color_continuous_scale=colorscale if color_col is not None else None,
        title=title,
    )

    fig.update_layout(width=width, height=height, font=dict(size=font_size))

    # Download the sunburst graph as an HTML file
    if download_path is not None:
        fig.write_html(download_path)

    if show:
        fig.show()

    return fig


def pie_plot(df, values_column, names_column, title='Pie Chart', width=800, height=600, filename='pie_chart.html',auto_open=False):
    """
    The pie_plot function creates a pie chart based on data from a pandas DataFrame using Plotly. The function takes the following parameters:

    df: A pandas DataFrame containing the data to be plotted.
    values_column: The name of the DataFrame column containing the values for the pie chart slices.
    names_column: The name of the DataFrame column containing the names for the pie chart slices.
    title: The title of the chart. Default is 'Pie Chart'.
    width: The width of the chart in pixels. Default is 800.
    height: The height of the chart in pixels. Default is 600.
    filename: The name of the file to save the chart to. Default is 'pie_chart.html'.

    The function returns a Plotly figure object.
    """
    import plotly.express as px
    import plotly.io as pio


    fig = px.pie(df, values=values_column, names=names_column, title=title)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(width=width, height=height)
    pio.write_html(fig, file=filename, auto_open=auto_open)
    return fig


# ==================== ADVANCED VISUALIZATIONS ====================

def sankey(df, source_col, target_col, value_col,
           title='Sankey Diagram',
           color_col=None,
           width=1000, height=600,
           filename=None, auto_open=False,
           show=True):
    """
    Create a Sankey diagram to visualize flows between categories.

    Perfect for: Customer journeys, budget allocation, process flows, conversion funnels.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the flow data.
    source_col : str
        Column name for source nodes.
    target_col : str
        Column name for target nodes.
    value_col : str
        Column name for flow values/weights.
    title : str, optional
        Title of the diagram.
    color_col : str, optional
        Column to use for coloring flows.
    width : int, optional
        Width in pixels.
    height : int, optional
        Height in pixels.
    filename : str, optional
        Path to save the HTML file.
    auto_open : bool, optional
        Whether to open the file in browser.

    Returns:
    --------
    plotly.graph_objs.Figure
        The Sankey diagram figure.

    Examples:
    ---------
    >>> # Customer journey visualization
    >>> df = pd.DataFrame({
    ...     'from': ['Homepage', 'Homepage', 'Product', 'Product', 'Cart'],
    ...     'to': ['Product', 'Exit', 'Cart', 'Exit', 'Purchase'],
    ...     'count': [1000, 200, 600, 100, 450]
    ... })
    >>> sankey(df, 'from', 'to', 'count', title='User Journey')

    >>> # Budget allocation
    >>> df = pd.DataFrame({
    ...     'category': ['Revenue', 'Revenue', 'Marketing', 'Operations'],
    ...     'subcategory': ['Marketing', 'Operations', 'Digital', 'Salaries'],
    ...     'amount': [500000, 300000, 300000, 200000]
    ... })
    >>> sankey(df, 'category', 'subcategory', 'amount', title='Budget Flow')
    """
    import plotly.graph_objs as go
    import os
    import pandas as pd

    # Get unique nodes
    all_nodes = pd.concat([df[source_col], df[target_col]]).unique()
    node_dict = {node: idx for idx, node in enumerate(all_nodes)}

    # Map sources and targets to indices
    sources = df[source_col].map(node_dict).tolist()
    targets = df[target_col].map(node_dict).tolist()
    values = df[value_col].tolist()

    # Colors
    if color_col:
        colors = df[color_col].tolist()
    else:
        # Generate color palette
        import plotly.express as px
        color_palette = px.colors.qualitative.Plotly
        colors = [color_palette[i % len(color_palette)] for i in range(len(all_nodes))]

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes.tolist(),
            color=colors if isinstance(colors[0], str) else None
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(0,0,0,0.2)'
        )
    )])

    fig.update_layout(
        title=title,
        font=dict(size=12),
        width=width,
        height=height
    )

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    if show:
        fig.show()
    return fig


def treemap(df, path_cols, value_col,
            color_col=None,
            title='Treemap',
            colorscale='Blues',
            width=1000, height=600,
            filename=None, auto_open=False,
            show=True):
    """
    Create a treemap for hierarchical data visualization.

    Perfect for: Market share, file systems, organizational structures, budget breakdowns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing hierarchical data.
    path_cols : list of str
        Column names defining the hierarchy (from top to bottom).
        Example: ['Country', 'State', 'City'] or ['Category', 'Subcategory', 'Product']
    value_col : str
        Column name for the values (determines rectangle size).
    color_col : str, optional
        Column to use for coloring rectangles.
    title : str, optional
        Title of the treemap.
    colorscale : str, optional
        Plotly colorscale name.
    width : int, optional
        Width in pixels.
    height : int, optional
        Height in pixels.
    filename : str, optional
        Path to save the HTML file.
    auto_open : bool, optional
        Whether to open the file in browser.

    Returns:
    --------
    plotly.graph_objs.Figure
        The treemap figure.

    Examples:
    ---------
    >>> # Sales by category hierarchy
    >>> df = pd.DataFrame({
    ...     'category': ['Electronics', 'Electronics', 'Clothing', 'Clothing'],
    ...     'subcategory': ['Phones', 'Laptops', 'Shirts', 'Pants'],
    ...     'product': ['iPhone', 'MacBook', 'T-Shirt', 'Jeans'],
    ...     'sales': [50000, 80000, 15000, 20000],
    ...     'profit': [15000, 25000, 8000, 10000]
    ... })
    >>> treemap(df, ['category', 'subcategory', 'product'], 'sales',
    ...         color_col='profit', title='Sales by Product Category')

    >>> # Market share visualization
    >>> df = pd.DataFrame({
    ...     'region': ['North America', 'North America', 'Europe', 'Europe'],
    ...     'country': ['USA', 'Canada', 'Germany', 'France'],
    ...     'market_share': [45, 15, 20, 15]
    ... })
    >>> treemap(df, ['region', 'country'], 'market_share')
    """
    import plotly.express as px
    import os

    fig = px.treemap(
        df,
        path=path_cols,
        values=value_col,
        color=color_col if color_col else value_col,
        color_continuous_scale=colorscale,
        title=title
    )

    fig.update_layout(
        width=width,
        height=height
    )

    fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percent: %{percentParent}<extra></extra>'
    )

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    if show:
        fig.show()
    return fig


def violin(df, x_col=None, y_col=None,
           color_col=None,
           title='Violin Plot',
           box_visible=True,
           meanline_visible=True,
           width=800, height=600,
           template='plotly_white',
           filename=None, auto_open=False,
           show=True):
    """
    Create violin plots for comparing distributions across categories.

    Combines box plot and density plot - shows distribution shape, quartiles, and outliers.
    Perfect for: A/B testing results, comparing groups, salary distributions by department.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    x_col : str, optional
        Column for x-axis (categorical variable).
    y_col : str
        Column for y-axis (continuous variable).
    color_col : str, optional
        Column to use for coloring violins.
    title : str, optional
        Title of the plot.
    box_visible : bool, optional
        Show box plot inside violin.
    meanline_visible : bool, optional
        Show mean line.
    width : int, optional
        Width in pixels.
    height : int, optional
        Height in pixels.
    template : str, optional
        Plotly template.
    filename : str, optional
        Path to save the HTML file.
    auto_open : bool, optional
        Whether to open the file in browser.

    Returns:
    --------
    plotly.graph_objs.Figure
        The violin plot figure.

    Examples:
    ---------
    >>> # Compare salaries by department
    >>> df = pd.DataFrame({
    ...     'department': ['Engineering']*100 + ['Sales']*100 + ['Marketing']*100,
    ...     'salary': np.random.normal(100000, 20000, 100).tolist() +
    ...               np.random.normal(80000, 15000, 100).tolist() +
    ...               np.random.normal(75000, 12000, 100).tolist()
    ... })
    >>> violin(df, x_col='department', y_col='salary',
    ...        title='Salary Distribution by Department')

    >>> # A/B test results
    >>> df = pd.DataFrame({
    ...     'variant': ['A']*200 + ['B']*200,
    ...     'conversion_time': np.random.exponential(30, 200).tolist() +
    ...                        np.random.exponential(25, 200).tolist()
    ... })
    >>> violin(df, 'variant', 'conversion_time',
    ...        title='Conversion Time: A vs B')
    """
    import plotly.express as px
    import os

    fig = px.violin(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        box=box_visible,
        title=title,
        template=template
    )

    if meanline_visible:
        fig.update_traces(meanline_visible=True)

    fig.update_layout(
        width=width,
        height=height
    )

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    if show:
        fig.show()
    return fig


def scatter_3d(df, x_col, y_col, z_col,
               color_col=None, size_col=None,
               title='3D Scatter Plot',
               hover_data=None,
               width=900, height=700,
               template='plotly_white',
               filename=None, auto_open=False,
               show=True):
    """
    Create 3D scatter plot for multi-dimensional data exploration.

    Perfect for: Cluster visualization, feature relationships, dimensionality reduction results.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    x_col : str
        Column for x-axis.
    y_col : str
        Column for y-axis.
    z_col : str
        Column for z-axis.
    color_col : str, optional
        Column to use for coloring points.
    size_col : str, optional
        Column to use for sizing points.
    title : str, optional
        Title of the plot.
    hover_data : list, optional
        Additional columns to show on hover.
    width : int, optional
        Width in pixels.
    height : int, optional
        Height in pixels.
    template : str, optional
        Plotly template.
    filename : str, optional
        Path to save the HTML file.
    auto_open : bool, optional
        Whether to open the file in browser.

    Returns:
    --------
    plotly.graph_objs.Figure
        The 3D scatter plot figure.

    Examples:
    ---------
    >>> # Visualize clusters in 3D space
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=300, centers=4, n_features=3, random_state=42)
    >>> df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])
    >>> df['cluster'] = y
    >>> scatter_3d(df, 'feature_1', 'feature_2', 'feature_3',
    ...            color_col='cluster', title='3D Cluster Visualization')

    >>> # Product analysis: price, quality, sales
    >>> df = pd.DataFrame({
    ...     'price': np.random.uniform(10, 100, 200),
    ...     'quality_score': np.random.uniform(1, 10, 200),
    ...     'sales_volume': np.random.uniform(100, 1000, 200),
    ...     'category': np.random.choice(['A', 'B', 'C'], 200)
    ... })
    >>> scatter_3d(df, 'price', 'quality_score', 'sales_volume',
    ...            color_col='category', size_col='sales_volume')
    """
    import plotly.express as px
    import os

    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        size=size_col,
        hover_data=hover_data,
        title=title,
        template=template
    )

    fig.update_layout(
        width=width,
        height=height,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        )
    )

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    if show:
        fig.show()
    return fig


def animated_scatter(df, x_col, y_col,
                    animation_frame,
                    color_col=None, size_col=None,
                    title='Animated Scatter Plot',
                    range_x=None, range_y=None,
                    hover_name=None,
                    width=900, height=600,
                    template='plotly_white',
                    filename=None, auto_open=False,
                    show=True):
    """
    Create animated scatter plot to show evolution over time or parameter changes.

    Perfect for: Time-series, parameter sensitivity analysis, evolution tracking.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    x_col : str
        Column for x-axis.
    y_col : str
        Column for y-axis.
    animation_frame : str
        Column to use for animation frames (e.g., 'year', 'iteration', 'time').
    color_col : str, optional
        Column to use for coloring points.
    size_col : str, optional
        Column to use for sizing points.
    title : str, optional
        Title of the plot.
    range_x : list, optional
        Fixed x-axis range [min, max].
    range_y : list, optional
        Fixed y-axis range [min, max].
    hover_name : str, optional
        Column to show as hover label.
    width : int, optional
        Width in pixels.
    height : int, optional
        Height in pixels.
    template : str, optional
        Plotly template.
    filename : str, optional
        Path to save the HTML file.
    auto_open : bool, optional
        Whether to open the file in browser.

    Returns:
    --------
    plotly.graph_objs.Figure
        The animated scatter plot figure.

    Examples:
    ---------
    >>> # Country GDP over time
    >>> df = pd.DataFrame({
    ...     'country': ['USA', 'China', 'India'] * 10,
    ...     'year': list(range(2010, 2020)) * 3,
    ...     'gdp': np.random.uniform(10000, 50000, 30),
    ...     'population': np.random.uniform(100, 1400, 30)
    ... })
    >>> animated_scatter(df, 'gdp', 'population', 'year',
    ...                  color_col='country', hover_name='country',
    ...                  title='GDP vs Population Over Time')

    >>> # Machine learning training progress
    >>> df = pd.DataFrame({
    ...     'epoch': list(range(1, 101)),
    ...     'train_loss': np.exp(-np.linspace(0, 3, 100)) + np.random.normal(0, 0.02, 100),
    ...     'val_loss': np.exp(-np.linspace(0, 2.8, 100)) + np.random.normal(0, 0.03, 100),
    ...     'learning_rate': [0.001] * 100
    ... })
    >>> animated_scatter(df, 'train_loss', 'val_loss', 'epoch',
    ...                  title='Training Progress Animation')
    """
    import plotly.express as px
    import os

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        animation_frame=animation_frame,
        color=color_col,
        size=size_col,
        hover_name=hover_name,
        title=title,
        range_x=range_x,
        range_y=range_y,
        template=template
    )

    # Smooth animation
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 50

    fig.update_layout(
        width=width,
        height=height
    )

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    if show:
        fig.show()
    return fig


def animated_line(df, x_col, y_col,
                 animation_frame,
                 color_col=None,
                 title='Animated Line Plot',
                 range_x=None, range_y=None,
                 width=900, height=600,
                 template='plotly_white',
                 filename=None, auto_open=False,
                 show=True):
    """
    Create animated line plot to show time-series evolution.

    Perfect for: Stock prices over time, metrics evolution, growth trends.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data (must be sorted by animation_frame).
    x_col : str
        Column for x-axis.
    y_col : str
        Column for y-axis.
    animation_frame : str
        Column to use for animation frames.
    color_col : str, optional
        Column to differentiate multiple lines.
    title : str, optional
        Title of the plot.
    range_x : list, optional
        Fixed x-axis range [min, max].
    range_y : list, optional
        Fixed y-axis range [min, max].
    width : int, optional
        Width in pixels.
    height : int, optional
        Height in pixels.
    template : str, optional
        Plotly template.
    filename : str, optional
        Path to save the HTML file.
    auto_open : bool, optional
        Whether to open the file in browser.

    Returns:
    --------
    plotly.graph_objs.Figure
        The animated line plot figure.

    Examples:
    ---------
    >>> # Stock price animation
    >>> dates = pd.date_range('2020-01-01', periods=365, freq='D')
    >>> df = pd.DataFrame({
    ...     'date': dates,
    ...     'price': 100 + np.cumsum(np.random.randn(365)),
    ...     'month': dates.to_period('M').astype(str)
    ... })
    >>> animated_line(df, 'date', 'price', 'month',
    ...               title='Stock Price Evolution')
    """
    import plotly.express as px
    import os

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        animation_frame=animation_frame,
        color=color_col,
        title=title,
        range_x=range_x,
        range_y=range_y,
        template=template
    )

    fig.update_layout(
        width=width,
        height=height
    )

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    if show:
        fig.show()
    return fig


def funnel_chart(df, stage_col, value_col,
                title='Funnel Chart',
                color_col=None,
                width=800, height=600,
                filename=None, auto_open=False,
                show=True):
    """
    Create funnel chart to visualize conversion processes.

    Perfect for: Sales funnels, conversion pipelines, user journey drop-offs.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing funnel data (should be sorted by stage).
    stage_col : str
        Column containing stage names.
    value_col : str
        Column containing values for each stage.
    title : str, optional
        Title of the chart.
    color_col : str, optional
        Column to use for coloring stages.
    width : int, optional
        Width in pixels.
    height : int, optional
        Height in pixels.
    filename : str, optional
        Path to save the HTML file.
    auto_open : bool, optional
        Whether to open the file in browser.

    Returns:
    --------
    plotly.graph_objs.Figure
        The funnel chart figure.

    Examples:
    ---------
    >>> # Sales funnel
    >>> df = pd.DataFrame({
    ...     'stage': ['Awareness', 'Interest', 'Consideration', 'Purchase', 'Loyalty'],
    ...     'users': [10000, 5000, 2000, 800, 300]
    ... })
    >>> funnel_chart(df, 'stage', 'users', title='Sales Funnel')

    >>> # Website conversion funnel
    >>> df = pd.DataFrame({
    ...     'stage': ['Visit', 'Sign Up', 'Trial', 'Subscribe'],
    ...     'count': [50000, 10000, 3000, 1500],
    ...     'conversion_rate': [100, 20, 6, 3]
    ... })
    >>> funnel_chart(df, 'stage', 'count',
    ...              title='Website Conversion Funnel')
    """
    import plotly.graph_objs as go
    import os

    # Calculate conversion rates
    values = df[value_col].tolist()
    stages = df[stage_col].tolist()

    # Calculate percentages
    if len(values) > 0:
        initial = values[0]
        percentages = [(v / initial * 100) for v in values]
        text = [f"{stage}<br>{val:,} ({pct:.1f}%)"
                for stage, val, pct in zip(stages, values, percentages)]
    else:
        text = stages

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="text",
        text=text,
        marker=dict(
            color=df[color_col].tolist() if color_col else None
        )
    ))

    fig.update_layout(
        title=title,
        width=width,
        height=height
    )

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    if show:
        fig.show()
    return fig


def waterfall_chart(df, category_col, value_col,
                   title='Waterfall Chart',
                   measure=None,
                   width=900, height=600,
                   filename=None, auto_open=False,
                   show=True):
    """
    Create waterfall chart to show cumulative effect of sequential values.

    Perfect for: Financial statements, profit/loss breakdown, sequential changes.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    category_col : str
        Column containing categories/labels.
    value_col : str
        Column containing values (positive or negative).
    title : str, optional
        Title of the chart.
    measure : list, optional
        List indicating measurement type for each category: 'relative', 'absolute', or 'total'.
        If None, auto-detects based on values.
    width : int, optional
        Width in pixels.
    height : int, optional
        Height in pixels.
    filename : str, optional
        Path to save the HTML file.
    auto_open : bool, optional
        Whether to open the file in browser.

    Returns:
    --------
    plotly.graph_objs.Figure
        The waterfall chart figure.

    Examples:
    ---------
    >>> # Profit breakdown
    >>> df = pd.DataFrame({
    ...     'category': ['Revenue', 'COGS', 'Marketing', 'Operations', 'Net Profit'],
    ...     'value': [1000000, -400000, -150000, -200000, 250000]
    ... })
    >>> waterfall_chart(df, 'category', 'value',
    ...                 title='Profit & Loss Breakdown')

    >>> # Cash flow analysis
    >>> df = pd.DataFrame({
    ...     'item': ['Starting Cash', 'Sales', 'Expenses', 'Investment', 'Ending Cash'],
    ...     'amount': [100000, 50000, -30000, -20000, 100000],
    ...     'type': ['absolute', 'relative', 'relative', 'relative', 'total']
    ... })
    >>> waterfall_chart(df, 'item', 'amount', measure=df['type'].tolist())
    """
    import plotly.graph_objs as go
    import os

    categories = df[category_col].tolist()
    values = df[value_col].tolist()

    # Auto-detect measure types if not provided
    if measure is None:
        measure = ['relative'] * len(categories)
        if len(categories) > 0:
            measure[0] = 'absolute'  # First is usually starting point
            measure[-1] = 'total'    # Last is usually total

    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=measure,
        x=categories,
        textposition="outside",
        text=[f"{v:+,.0f}" if v != 0 else "" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        width=width,
        height=height,
        xaxis_title=category_col,
        yaxis_title=value_col
    )

    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        filepath = os.path.join(os.getcwd(), filename)
        fig.write_html(filepath, auto_open=auto_open)

    if show:
        fig.show()
    return fig


# ==================== DASHBOARD TEMPLATE ====================

def _run_dash_app(app, host, port, debug):
    """Run a Dash app across old and new Dash versions."""
    runner = getattr(app, 'run', None) or getattr(app, 'run_server', None)
    if runner is None:
        raise AttributeError("Dash app does not expose run() or run_server().")

    try:
        return runner(host=host, port=port, debug=debug)
    except TypeError:
        return runner(port=port, debug=debug)


def create_dashboard_template(title='Data Analysis Dashboard',
                             port=8050,
                             debug=True,
                             run=False,
                             host='127.0.0.1'):
    """
    Create a basic Plotly Dash dashboard template.

    Returns a configured Dash app that you can customize with your own callbacks and layouts.

    Parameters:
    -----------
    title : str, optional
        Dashboard title.
    port : int, optional
        Port to run the dashboard on.
    debug : bool, optional
        Run in debug mode.
    run : bool, optional
        Start the Dash server immediately. This blocks the current cell/process
        until the server is stopped.
    host : str, optional
        Host interface used when run=True.

    Returns:
    --------
    dash.Dash
        Configured Dash application.

    Examples:
    ---------
    >>> # Basic usage
    >>> app = create_dashboard_template('Sales Dashboard')
    >>>
    >>> # Add your custom layout and callbacks
    >>> app.layout = html.Div([
    ...     html.H1('My Dashboard'),
    ...     dcc.Graph(id='my-graph', figure=my_figure)
    ... ])
    >>>
    >>> app.run(port=8050)

    >>> # Or start the server immediately
    >>> app = create_dashboard_template('Sales Dashboard', run=True)

    >>> # Full example with callbacks
    >>> app = create_dashboard_template('Interactive Dashboard')
    >>>
    >>> app.layout = html.Div([
    ...     html.H1('Sales Analysis'),
    ...     dcc.Dropdown(
    ...         id='category-dropdown',
    ...         options=[{'label': cat, 'value': cat} for cat in df['category'].unique()],
    ...         value=df['category'].unique()[0]
    ...     ),
    ...     dcc.Graph(id='sales-graph')
    ... ])
    >>>
    >>> @app.callback(
    ...     Output('sales-graph', 'figure'),
    ...     Input('category-dropdown', 'value')
    ... )
    >>> def update_graph(selected_category):
    ...     filtered_df = df[df['category'] == selected_category]
    ...     fig = px.bar(filtered_df, x='product', y='sales')
    ...     return fig
    >>>
    >>> app.run(port=8050)
    """
    try:
        from dash import Dash, html
    except ImportError:
        raise ImportError(
            "Dash is required for dashboard creation. "
            "Install it with: pip install dash"
        )

    # Initialize the app
    app = Dash(__name__, title=title)

    # Default layout (users can override this)
    app.layout = html.Div([
        html.H1(title, style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Hr(),
        html.Div([
            html.P(
                "This is a dashboard template. Customize the layout by setting app.layout "
                "and add callbacks for interactivity.",
                style={'textAlign': 'center', 'fontSize': '14px', 'color': '#7f8c8d'}
            )
        ], style={'padding': '20px'})
    ])

    print(f"Dashboard template created. Run with: app.run(port={port})")
    print(f"Access at: http://127.0.0.1:{port}/")

    if run:
        print(f"Starting dashboard at: http://{host}:{port}/")
        _run_dash_app(app, host=host, port=port, debug=debug)

    return app


def quick_dashboard(df, title='Quick Dashboard',
                   numerical_cols=None,
                   categorical_cols=None,
                   port=8050,
                   run=False,
                   debug=True,
                   host='127.0.0.1'):
    """
    Create a quick interactive dashboard from a DataFrame with automatic visualizations.

    Generates a multi-page dashboard with:
    - Overview page with summary statistics
    - Distribution plots for numerical columns
    - Category analysis for categorical columns
    - Correlation heatmap
    - Interactive filtering

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to visualize.
    title : str, optional
        Dashboard title.
    numerical_cols : list, optional
        Numerical columns to include (auto-detected if None).
    categorical_cols : list, optional
        Categorical columns to include (auto-detected if None).
    port : int, optional
        Port to run the dashboard on.
    run : bool, optional
        Start the Dash server immediately. This blocks the current cell/process
        until the server is stopped.
    debug : bool, optional
        Run in debug mode when run=True.
    host : str, optional
        Host interface used when run=True.

    Returns:
    --------
    dash.Dash
        Configured and ready-to-run Dash application.

    Examples:
    ---------
    >>> # Quick dashboard from any DataFrame
    >>> app = quick_dashboard(df, title='Sales Analysis Dashboard')
    >>> app.run(port=8050)

    >>> # Or start it immediately
    >>> app = quick_dashboard(df, title='Sales Analysis Dashboard', run=True)

    >>> # With specific columns
    >>> app = quick_dashboard(
    ...     df,
    ...     title='Product Analysis',
    ...     numerical_cols=['price', 'sales', 'profit'],
    ...     categorical_cols=['category', 'region']
    ... )
    >>> app.run(debug=True)
    """
    try:
        from dash import Dash, html, dcc, Input, Output
        import plotly.express as px
        import plotly.graph_objs as go
    except ImportError:
        raise ImportError(
            "Dash is required for dashboard creation. "
            "Install it with: pip install dash"
        )

    # Auto-detect column types
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Limit to reasonable cardinality
        categorical_cols = [col for col in categorical_cols
                           if df[col].nunique() < 50]

    # Initialize app
    app = Dash(__name__, title=title)

    # Create layout
    app.layout = html.Div([
        html.H1(title, style={
            'textAlign': 'center',
            'color': '#2c3e50',
            'marginBottom': '30px'
        }),

        # Summary statistics
        html.Div([
            html.H2('Dataset Overview', style={'color': '#34495e'}),
            html.Div([
                html.Div([
                    html.H3(f"{len(df):,}", style={'color': '#3498db', 'marginBottom': '5px'}),
                    html.P('Total Rows', style={'color': '#7f8c8d'})
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px'}),

                html.Div([
                    html.H3(f"{len(df.columns)}", style={'color': '#e74c3c', 'marginBottom': '5px'}),
                    html.P('Columns', style={'color': '#7f8c8d'})
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px'}),

                html.Div([
                    html.H3(f"{df.isnull().sum().sum():,}", style={'color': '#f39c12', 'marginBottom': '5px'}),
                    html.P('Missing Values', style={'color': '#7f8c8d'})
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px'}),

                html.Div([
                    html.H3(f"{df.duplicated().sum():,}", style={'color': '#9b59b6', 'marginBottom': '5px'}),
                    html.P('Duplicates', style={'color': '#7f8c8d'})
                ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '40px'})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),

        # Filters
        html.Div([
            html.H3('Filters', style={'color': '#34495e'}),
            html.Div([
                html.Label('Select Numerical Column:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='num-col-dropdown',
                    options=[{'label': col, 'value': col} for col in numerical_cols],
                    value=numerical_cols[0] if numerical_cols else None,
                    style={'marginBottom': '20px'}
                ),

                html.Label('Select Categorical Column:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='cat-col-dropdown',
                    options=[{'label': col, 'value': col} for col in categorical_cols],
                    value=categorical_cols[0] if categorical_cols else None
                ),
            ], style={'padding': '20px'})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),

        # Visualizations
        html.Div([
            # Distribution plot
            html.Div([
                dcc.Graph(id='distribution-plot')
            ], style={'width': '48%', 'display': 'inline-block'}),

            # Category plot
            html.Div([
                dcc.Graph(id='category-plot')
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        ]),

        html.Div([
            # Correlation heatmap
            dcc.Graph(id='correlation-heatmap')
        ], style={'marginTop': '30px'}),

    ], style={'padding': '40px', 'fontFamily': 'Arial, sans-serif'})

    # Callbacks
    @app.callback(
        Output('distribution-plot', 'figure'),
        Input('num-col-dropdown', 'value')
    )
    def update_distribution(selected_col):
        if selected_col is None:
            return go.Figure()

        fig = px.histogram(
            df,
            x=selected_col,
            title=f'Distribution of {selected_col}',
            template='plotly_white'
        )
        fig.update_layout(showlegend=False)
        return fig

    @app.callback(
        Output('category-plot', 'figure'),
        Input('cat-col-dropdown', 'value')
    )
    def update_category(selected_col):
        if selected_col is None:
            return go.Figure()

        value_counts = df[selected_col].value_counts().head(10)
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Top 10 {selected_col}',
            labels={'x': selected_col, 'y': 'Count'},
            template='plotly_white'
        )
        return fig

    @app.callback(
        Output('correlation-heatmap', 'figure'),
        Input('num-col-dropdown', 'value')  # Just for triggering
    )
    def update_heatmap(_):
        if not numerical_cols:
            return go.Figure()

        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title='Correlation Heatmap',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f'
        )
        return fig

    print(f"\n{'='*60}")
    print("Quick Dashboard Created Successfully!")
    print(f"{'='*60}")
    print(f"Access at: http://127.0.0.1:{port}/")
    print("Press Ctrl+C to stop the server")
    print(f"{'='*60}\n")

    if run:
        _run_dash_app(app, host=host, port=port, debug=debug)

    return app
