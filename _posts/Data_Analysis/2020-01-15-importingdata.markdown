---
layout: post
title:  " Importing data in Python using Pandas"
date:   2020-01-15 15:23:09 -0500
categories: data analysis
permalink: /data-analysis/importingdata.html
---
Pandas is one of the most preferred and widely used tools in Python for data analysis(Thank Wes McKinney, the developer of Pandas). Pandas brings the functionality of Excel together with the power of the Python language. Excel columns become pandas Series, tables become DataFrames, and complex formulas become Python functions. There are three very important components when we begin the data analysis:  

1. Importing data
2. Manipulating data
3. Visualizing data

The three components are writen in three posts: Importing data in Python using Pandas, Working with Pandas Dataframes and data visualization(part1:overview, part2:application). Importing data is one of the most essential and very first steps in any data related problem. The ability to import the data correctly is a must-have skill for every data scientist. Pandas has an input and output API which has a set of top-level reader and writer methods. Below are Pandas's method for loading and writing different data format, which means if your data is in any of the below forms, you can use pandas to load that data format and even write into a particular format.

![image](https://user-images.githubusercontent.com/54314187/76475442-411fac00-63d5-11ea-9f91-dc8742a270f0.png)

In this post, I will summarize some practical skills for data importing with the data format of CSV, Exel(xlsx), HTML table, json or SQL. 

## csv files

Here is the overview of how to use Pandas to load CSV to dataframes and how to write dataframes to CSV.

- Load CSV files to dataframe using  Pandas read_csv  
  - locally  

        df = pd.read_csv('yourCSVfile.csv')  

  - from the Web  
  It’s very simple we just put the URL in as the first parameter in the read_csv method.  

        df = pd.read_csv('url')

- Remove unnamed columns  
we often see that we get a column named ‘Unnamed: 0’. we can set this column as index column by index_col parameter or delete that column by drop() method. 

        df = pd.read_csv('url', index_col=0)  

        df1.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)  

- Read certain columns  
In some cases, we don’t want to parse every column in the CSV file. To only read certain columns we can use the parameter usecols.  

        df = pd.read_csv('yourCSVfile.csv', usecols=[a, b, c, d])  
        or  
        cols = pd.read_csv('url', nrows=1).columns  
        df = pd.read_csv('url', usecols=cols[1:])  

- Skipping rows and reading certain rows  
Here’s the Pandas read_csv example with skipping the three first rows: 

        df = pd.read_csv('yourCSVfile', skiprows=3)  
        or  
        df = pd.read_csv('yourCSVfile', header=3) 
    
If we don’t want to read every row in the CSV file we ca use the parameter nrows.  Below is to read the first 10 rows of a CSV file.
        df = pd.read_csv('yourCSVfile', nrows=10)  

- Reading many CSV files  
I found using Python glob module and the list comprehension is a simple way to read many CSV files.

        import glob  
        csv_files = glob.glob('files/*.csv')  
        dfs = [pd.read_csv(csv_file) for csv_file in csv_files]   
        df = pd.concat(dfs, sort=False)  

- Saving dataframes to CSV using Pandas to_csv  
It’s quite simple to write the dataframe to CSV file using Pandas to_csv method.  We will get a new column when we are not using any parameters. This column is the index column from our Pandas dataframe. When working with Pandas to_csv, we can use the parameter index and set it to False to get rid of that column.  

        df.to_csv('filename.csv', index=False)

##  Excel Files

I am going to exhibit how to read Excel files and spreadsheets to Pandas dataframe objects using read_excel() method, which works well both for importing local files and from a URL to a dataframe.

- Read Excel files and Spreadsheets using read_excel 

        df = pd.read_excel('yourExcelfile.xlsx') 
     
If we don’t pass any other parameters, such as sheet name, it will read the first sheet in the excel file. We can set the parameter sheet_name for loading specific sheet. for example  

        df = pd.read_excel('yourExcelfile.xlsx', sheet_name=1)  
   
- Merging many sheets to a dataframe 

        df = pd.read_excel('yourExcelfile.xlsx', sheet_name=['sheet1','sheet2'])  

or set sheet_name=None for all sheets.  

        df = pd.read_excel('yourExcelfile.xlsx', sheet_name=None) 
    
By using the parameter sheet_name, and a list of names, we will get an ordered dictionary containing two dataframes. we may join the data from all sheets.  we just use the concat function and loop over the keys (i.e., sheets) for Merging all the data as one datafrme.

        df1 = pd.concat(df[subframe] for subframe in df.keys())  
    
- Write a dataframe to an Excel file

        df = pd.to_excel('Excelfile.xlsx', sheet_name='sheet1', index=False)  
     
- Taking many dataframes and writing them to one Excel file with many sheets 
we are going to use Pandas ExcelWriter and Pandas to_excel to write multiple Pandas dataframes to one Excel file. That is if we have many dataframes(for example: df1, df2, df3) that we want to store in one Excel file but on different sheets, we need to use Pandas ExcelWriter now:

        dfs = {'sheet1':df1, 'sheet2':df2, 'sheet3':df3}   
        writer = pd.ExcelWriter('Excelfile.xlsx', engine='xlsxwriter')   
        for sheet_name in dfs.keys():   
        dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)   
        writer.save()  
    
## html table
We generally think we need requests and BeautifulSoup to parse HTML tables from website. Acturally pandas has the ability to parse the HTML table with a very simple way pd.read_html(). The function read_html always returns a list of DataFrame objects.  we need to pass the row to use as header such as dfs[1] for the fist table, dfs[3] getting the third table.

    url = 'https://www.basketball-reference.com/draft/NBA_2019.html' 
    dfs = pd.read_html(url, header=1)
    
    len(dfs)  
    out: 3
    
    dfs  

![image](https://user-images.githubusercontent.com/54314187/76479581-a1691a80-63e2-11ea-90f8-9039b6420625.png)

    
    df = dfs[0]  
    df.head()  
    
![image](https://user-images.githubusercontent.com/54314187/76475988-f3a43e80-63d6-11ea-8738-df1ad43f59f9.png)

## json file  
JSON, short for JavaScript Object Notation, is a compact, text based format used to exchange data. This format that is common for downloading, and storing, information from web servers via so-called Web APIs. JSON is a text-based format and  when opening up a JSON file, we will recognize the structure. That is, it is not so different from Python’s structure for a dictionary.  We can import json file by using pandas read_json method.

    df = pd.read_json('yourJSONonfile.json')
    
Note that sometimes using Pandas read_json seems to create a dataframe with dictionaries within each cell. These data nested within dictionaries have to work with the Python module Request.  


## Importing from SQL  

Before I get the right answer for importing data from the database to pandas, it looks complicated.  There are several different database platforms, such as MySQL, SQL Server, PostgreSQL or Oracle. It also needs to get to know the SQL syntax for querying the table from database, and needs to know how to get connection with the database. Finally I figured out there is a very simpe but powerful method by using sqlalchemy package.

    import sqlalchemy as db  
    engine = db.create_engine('postgresql://postgres:password@localhost/dvdrental')  
    sql = """  
    SELECT *  
    FROM actor  
    """  
    df = pd.read_sql_query(sql, engine)  
    df.head()



In summary, I have focused on how to import data into Pandas dataframe. These might be useful when I want to advance my knowledge. That is deeply understanding to load data from different sources and different formats such as csv, excel, JSON, HTML tables and SQL.