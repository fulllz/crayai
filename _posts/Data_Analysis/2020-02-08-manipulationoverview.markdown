---
layout: post
title:  "  Overview of Data Manipulation and Visualization with Python Packages"
date:   2020-02-08 15:23:09 -0500
categories: data analysis
permalink: /data-analysis/manipulationoverview.html
---

There are three very important components when we begin the data analysis:  

1. Importing data
2. Manipulating data
3. Visualizing data

I did some work on [How importing data to Pandas](https://crayai.com/data-analysis/importingdata.html). Then I will focus on the overview of data manipulation and visualization with python packages in this post.

## Overview of data manipulation using Pandas

When we prepare data for analysis, data almost never comes in clean. We should diagnose our data for problems, extract, filter, and transform real-world datasets for analysis, analyse basic trends and patterns. I am going to summarize some common tasks that I think it is important to use pandas to add to the power of Python.

### Common data problems

- Inconsistent column names: inconsistent capitalization/bad characters.
- Need to process columns.
- Column types can signal unexpected data values
- Missing data: need to be identified and addressed.
- Outliers: need to investigate.
- Duplicate rows: found and dropped.
- Untidy

The python package Pyjanitor extends Pandas with a verb-based API. Pyjanitor can do  several things in one step such as adding columns, removing empty columns, droping missing data, and cleaning the column names. We all konw data scientist spends 80% time on wrangling data. it is a cool tool to speed the data cleaning process.

### Data types

One of the first steps when exploring a new data set is making sure the data types are set correctly. A possible confusing point about pandas data types is that there is some overlap between pandas, python and numpy. This table summarizes the key points:  

In order to convert data types in pandas, there are three basic options:  

1. Use `astype()` to force an appropriate datatype
2. Use pandas functions such as `to_numeric()` or `to_datetime()`
3. using `np.where()` to convert the column to a boolean
4. Create a custom function to convert the data for more complicated cases.

### Numerical data

Generally we will do statistical exploratory data analysis for the numerical data. Two aspects we should pay attention to: 

1. Frequency counts: Count the number of unique values in our data.
2. Summary statistics: find the general data distribution and outliers.

Pandas in python provides an interesting method `describe()`. The describe function applies basic statistical computations on the dataset like extreme values, count of data points standard deviation etc. Pandas provides the `cut()`and `qcut()` functions to breaking continuous values into discrete bins.

Pandas-profiling is one of modules can boost the statistical exploratory data analysis. The module offers out-of-the-box statistical profiling since the dataset we are using is tidy and standardize.  runing pandas_profiling on our data frame will return the Pandas Profiling Report, which includes overview, variables, correlations, missing values and sample.

### Text data

- Slicing strings 
Strings in a Series can be sliced using `str.slice()` method, or more conveniently, using brackets `str[]`.

- Capitalization of strings:  
`str.lower()`, `str.upper()`, `str.title()`.

- Checking for contents of a string  
`str.contains()`, `str.startswith()`, `str.endswith()`. 

- Regular expressions  

The re library for regular expressions is a formal way of specifying a pattern, sequence of characters, and pattern matching.  
Extract strings with a specific regex: `str.extract[r'[Aa-Zz]']`.  
Replace strings within a regex: `str.replace('Replace this', 'With this')`.  

Linking different record sets on text fields like names and addresses is a common but challenging data problem. Python provides two libraries that are useful for these types of problems and can support complex matching algorithms with a relatively simple API. The first one is called *fuzzymatcher* and provides an interface to link two pandas DataFrames together using probabilistic record linkage. The second option is the Python *recordlinkage* package which provides a robust set of tools to automate record linkage and perform data deduplication.

### Categorical data

In many practical Data Science activities, the data set will contain categorical variables. These variables are typically stored as text values which represent various traits. Some examples include color (“Gold”, “White”, “Blue”), weather (“sunny”, “cloudy”, “rainy”) etc. Converting categorical data to 'category' data type can make the DataFrame smaller in memory, and can make them be utilized by other Python libraries for analysis.

General guidelines for using categorical data types:

1. It is not necessary to convert all categorical data to the pandas category data type.
2. If the data set starts to approach an appreciable percentage of useable memory, then consider using categorical data types.
3. If we have very significant performance concerns with operations that are executed frequently, look at using categorical data.
4. Add some checks to make sure the data is clean and complete before converting to the pandas category type. Additionally, check for NaN values after combining or converting dataframes.

Regardless of what the value is used for, the challenge is determining how to use this data in the analysis. Therefore, we should figure out how to turn these text attributes into numerical values for further processing.  

- Find and Replace  
`df.replace()`

- Label Encoding 
convert a column to a category, then use those category values for label encoding by using the `cat.codes` accessor.  

- One Hot Encoding  
one-hot encoding is the process of converting categorical values into a 1-dimensional numerical vector. The basic strategy is to convert each category value into a new column and assigns a 1 or 0 (True/False) value to the column. Pandas supports this feature using `get_dummies()` method.

- Custom Binary Encoding 
Depending on the data set, we may use some combination of label encoding and one hot encoding to create a binary column that meets the needs for further analysis.  

### Datetime data

The pandas package is extremely useful for time series manipulation and analysis.  Specific objectives are to show you how to:  

- convert string data to a timestamp  

In pandas, a single point in time is represented as a Timestamp. We can use the `to_datetime()` function to create Timestamps from strings in a wide variety of date/time formats.

- index and slice the time series data in a data frame 

If we supply a list or array of strings as input to to_datetime(), it returns a sequence of date/time values in a DatetimeIndex object, which is the core data structure. The useful aspect of the DatetimeIndex is that the individual date/time components are all available as attributes such as year, month, day, and so on. `df.index.year, df.index.day, df.index.weekday_name`.   
Another very handy feature of pandas time series is partial-string indexing, where we can select all date/times which partially match a given string. For example, we can select the entire year 2019 with `df.loc['2019']`, or the entire month of December 2019 with `df.loc['2019-12']`.

- Resample the data at a different frequencies  
`df.resample('D').mean()`

- Roll windows  

Rolling windows split the data into time windows and the data in each window is aggregated with a function such as mean(), median(), sum(), etc.
`df.rolling(3).sum()`

- Time zones conversion.  
`real_t.tz_localize('UTC').tz_convert('US/Pacific')`.

### Indexing data

- Indexing   

Pandas supports indexing by using labels, position based integers or a list of boolean values (True/False). We can use square brackets `df[]` to select elements. In addition, there are two other ways:`df.loc[]` is label-based and `df.iloc[]` is position-based. Using a list of boolean values to select a row is called boolean indexing.

- Sampling the data 

If we have a large dataset, we might consider taking a sample of our data as an easy way `sample()`.

- Querying  

In order to inspect the data further by querying the data, we can use the `df.query()` method.

### Merging Dataframes

While merge, join, and concat all work to combine multiple DataFrames, A vertical combination would use a DataFrame’s concat method to combine the two DataFrames into a single DataFrame. The number of rows has increased but the number of columns has stayed the same.   
`df3 = pd.concat([df1, df2], ignore_index=True)`  

By contrast, the merge and join methods help to combine DataFrames horizontally. In this horizontal combination, we add columns to existing rows but don't add any additional row. The operation is similar to a JOIN operator in SQL. Types of merges have One-to-one, Many-to-one / one-to-many and Many-to-many. All use the same function, only difference is the DataFrames we are merging. Using either merge or join, we’ll need to specify how the DataFrames should be merged or joined. There are four possible values for how to join two DataFrames: 

1. Left: Retain all rows for the first DataFrame and enrich with columns from the second DataFrame where they match on the columns on which to join.
2. Right: Same as left but reversed, retain all rows for the second DataFrame and enrich with columns from matches in the first DataFrame. 
3. Inner: Retain only the intersection of the two DataFrames. 
4. Outer: Retain all rows from both DataFrames regardless of whether there are matching rows in the other DataFrame.
The syntax of `pd.merge()` method is:  
`df_merge = pd.merge(df1, df2, on= 'column_name', how ='left')`. 

### Reshaping data  

- Tidy data  

What is the tidy data: Columns represent separate variables, rows represent individual observations, and observational units form tables. Tidy data makes it easier to fix common data problems. Such as we can fix that columns containing values instead of variables by converting to tidy data. Why we want to reshape data? The reason is the choice that it is better for reporting or better for analysis.  We want to reshape it to another format enven if this table is already tidy. Because we might be interested in visually comparing. Or we might be interested in plotting the data.  

- pivot() method  

Let’s begin with looking at a table where the data is tidy. The format of this table can be referred to as the: 
stacked format-the individual observations are stacked on top of each other.  
record format-each row is a single record, i.e. a single observation.  
long format-this format will be long in the vertical direction as opposed to wide in the horizontal direction.  

Pivoting:
1. Turn unique values into separate columns. 
2. Convert analysis-friendly shape to reporting-friendly shape.  
3. Violates tidy data principle: rows contain observations. Multiple variables stored in the same column.   

In pandas, we can accomplish just that by using the pivot method of the dataframe. This produces a “pivot table”, which will be familiar to Excel users.  
`df.pivot(index='column1', columns='column2', values='column3')`

The format of this table can be referred to as: 
wide format-the table is now wider rather than longer.
unstacked format-the individual observations (one person/one date) are no longer stacked on top of each other.

- pivot_table 

Pandas dataframes also come with a pivot_table method, which has a parameter that species how to deal with duplicate values. Whenever you have duplicate values for one index/column pair, you need to use the pivot_table. 

`df_cookies.pivot_table(index='col1', columns='col2', values='col3', aggfunc='sum')`

- melt() method  

Melt in pandas reshapes dataframe from wide format to long format. It uses the id_vars[‘col_names’] for melt the dataframe by column names. In other words, we turned columns into rows in melting.  
`pd.melt(df, id_vars='col1', value_vars=['col2', 'col3', 'col4'])`

- Stacking and unstacking data 

In addition to the pivoting methods, pandas also has the two related concepts of stacking and unstacking data. These are primarily designed to operate on multi-indexed dataframes. With `df.stack()` and `df.unstack()`, we can toggle between hierarchical indices and hierarchical columns. In the case, we have a hierarchical index, so let’s see what unstack does. The operation moves one level of our hierarchical index to form a new level of columns in the dataframe. It producs a reshaped DataFrame with a new inner-most level of column labels.  To move back to a stacked format, we simple use stack. Stack method works with the MultiIndex objects in DataFrame, it returns a DataFrame with an index with a new inner-most level of row labels. It changes the wide table to a long table.  
`df_multi.unstack()`
`df.stack()`

### Grouping data

`groupby()` method is a split-apply-combine chained operation. Three steps: 

1. Split a table into group  
2. Apply some operations to each of those smaller tables  
3. Combine the results 

The typical sytax can be writen as: `df.groupby('column1')['column2'].mean()` 

Methods of a Pandas GroupBy object fall into a handful of categories:

- Aggregation methods--aggregating statisticcally about the data using such as sum, mean, or median. 

- Filter methods. This most commonly means using .filter() to drop entire groups based on some comparative statistic about that group and its sub-table. It also makes sense to include under this definition a number of methods that exclude particular rows from each group.  

- Transformation methods return a DataFrame with the same shape and indices as the original, but with different values.

- Meta methods are less concerned with the original object on which we called .groupby(), and more focused on giving you high-level information such as the number of groups and indices of those groups.  

- Plotting methods mimic the API of plotting for a Pandas Series or DataFrame, but typically break the output into multiple subplots.  

## Overview of data visualization 

Data visualization can vividly show the data and potential relation between features. Many amazing visualization libraries are available in python, which turns to be very versatile. Matplotlib is amazing and very flexible but need more coding. When I move on and try something new, I found two amazing libraries for data science:  
-Cufflinks  
-seaborn

## Seaborn

If matplotlib “tries to make easy things easy and hard things possible”, seaborn tries to make a well-defined set of hard things easy too.  Seaborn is a library for making statistical graphics in Python. It is built on top of matplotlib and closely integrated with pandas data structures. We can summarize as three kinds of plotting:

### Visualization for distribution

- distplot(), kdeplot(), rugplot()
- jointplot() & jointgrid
- pairplot() &pairgrid
- heatmap()

### Visualization for linear relation

- relplot()
- scatterplot(), lineplot()
- regplot, lmplot() &facetgrid
- residplot()
- jointplot() &pairplot() with kind parameters

### Visualization for category

- catplot()
- swarmplot() & stripplot()
- boxplot(), boxenplot & violinplot()
- barplot(), countplot() & pointplot()
- clustermap()

## Cufflinks

### Advantage of cufflinks  

Cufflinks is a wrapper around the plotly library specifically for plotting with Pandas dataframes. With cufflinks, we don't have to dig into the details of plotly, instead building our charts with minimal code. Basically, you can make charts directly in plotly for more control, or you can use cufflinks to rapidly prototype plots and explore the data. We can simply use the .iplot() method and specify the kind of chart we want to generate with the dataset. Basic graph includes:  

- bar chart  
The benefits of interactivity are that we can explore and subset the data as we like. 
- boxplot  
There’s a lot of information in a boxplot. Generating a box plot to demonstrate the shape of the distribution of each stat.
- scatterplot  
The scatterplot is found at the heart of most analyses - it allows us to see the evolution of a variable over time or the relationship between two (or more) variables.

### Advanced plots

There are some more plots that we probably won’t use all that often, but which can be quite impressive.

- heatmap
- pie charts
- geographic plotting--Choropleth
- 3D-plot

### More advanced plots:quant plotting

It becomes easy to plot candlestick graph combined volume. It is also very quick to create and modify interactive financial charts using Cufflinks and Pandas. 

## Summary 

Pandas is so powerful, and so meaningful. It has some challenge to understand the functionality for data analysis. Here I well organized with four kinds of data types(numerical data, text data, categorical data and datetime data) and four general operations(indexing, merging, reshaping and groupby data). Understanding specific data type with different method and Mastering the general manipulation of data provide a solid foundation in data science.  Catching beautiful packages of Seaborn and Cufflinks enhanced our ability for visualizing data analysis and results reporting.