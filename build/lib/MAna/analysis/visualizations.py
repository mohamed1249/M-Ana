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


def scatter(df, x_col, y_col, color_col=None, size_col=None, title=None, x_title=None, y_title=None, width=None, height=None, template='plotly_white', mode='markers', symbol='circle', opacity=0.7, marker=None, filename=None, auto_open=True):
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
        fig: The plotly figure object.
    """
    import os
    import plotly.express as px

    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, title=title, 
                     labels={x_col:x_title, y_col:y_title}, width=width, height=height, 
                     template=template, symbol=symbol, opacity=opacity, marker=marker, 
                     mode=mode, trendline='ols')
    
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


    l = len(df)*lenght/10 # count some lenght first,

    fig = make_subplots(rows=1, cols=1) # then creates a subplot and..

    fig.add_trace(go.Histogram(x=df[col], name=col)) # give it a histogram plot that shows the distripution,
    fig.add_trace(go.Line(x=[df[col].mean() for i in range(round(l))], y=list(range(round(l))), name=f'{col}\'s mean')) # a virtical line that shows where the mean is,
    fig.add_trace(go.Line(x=[df[col].median() for i in range(round(l))], y=list(range(round(l))), name=f'{col}\'s median')) # a virtical line that shows where the median is and
    fig.add_trace(go.Line(x=[df[col].mode()[0] for i in range(round(l))], y=list(range(round(l))), name=f'{col}\'s mode')) # a virtical line that shows where the mode is.
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
    import os
    import plotly.graph_objs as go

    download_path=os.getcwd()

    fig = go.Figure()
    fig.add_trace(go.Box(x=df[x_col], y=df[y_col], marker_color=color, name=y_col))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, width=width, height=height, showlegend=show_legend)
    fig.show()

    # Save the plot as a PNG file in the specified path
    filename = f"{y_col}_box_plot.png"
    filepath = os.path.join(download_path, filename)
    fig.write_image(filepath)


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
    return sns.heatmap(df.corr(), annot=True, cmap=cmap, linewidths=linewidths, annot_kws=annot_kws);


def pairplot(df, color=None, size=None):
    """
    A function to plot a pair plot for a given dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to plot the pair plot for.
    color : str or None, optional (default=None)
        A column name of df to use for color encoding the scatter plot matrix.
    size : str or None, optional (default=None)
        A column name of df to use for size encoding the scatter plot matrix.

    Returns:
    --------
    None

    Example:
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    >>> pairplot(df, color=iris.target, size=iris.target)
    """
    
    import plotly.express as px

    fig = px.scatter_matrix(df, color=color, size=size)
    fig.update_traces(diagonal_visible=False)
    fig.show()


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


def sunburst(df, hierarchy_cols, size_col, color_col=None, title=None, width=800, height=800, font_size=14, colorscale='YlOrRd', download_path=None):
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
    # Create a list of values for each hierarchy level
    import plotly.graph_objs as go

    hierarchy_values = []
    for i in range(len(hierarchy_cols)):
        hierarchy_values.append(df[hierarchy_cols[i]].unique().tolist())

    # Create a Plotly sunburst graph
    fig = go.Figure(go.Sunburst(
        labels=df[hierarchy_cols[-1]],
        parents=df[hierarchy_cols[-2]],
        values=df[size_col],
        branchvalues='total',
        marker=dict(
            colors=df[color_col] if color_col is not None else None,
            colorscale=colorscale
        ),
        textfont=dict(
            size=font_size
        ),
        insidetextorientation='radial',
        maxdepth=len(hierarchy_cols)-1
    ))

    # Set the colorbar title
    if color_col is not None:
        fig.update_layout(coloraxis_colorbar=dict(
            title=color_col.capitalize(),
            title_font=dict(
                size=font_size
            ),
            ticksuffix=' '
        ))

    # Set the sunburst graph title
    if title is not None:
        fig.update_layout(title={
            'text': title,
            'font': {
                'size': font_size
            }
        })

    # Set the sunburst graph dimensions
    fig.update_layout(width=width, height=height)

    # Download the sunburst graph as an HTML file
    if download_path is not None:
        fig.write_html(download_path)

    # Show the sunburst graph
    fig.show()


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