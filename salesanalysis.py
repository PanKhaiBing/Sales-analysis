### 1. How does customer age and gender influence their purchasing behavior?
### 2. Are there discernible patterns in sales across different time periods?
### 3. Which product categories hold the highest appeal among customers?
### 4. What insights can be gleaned from the distribution of product prices within each category?
### 5. What is the correlation between price, sales and quantity?

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('retailsales.csv')

### Data before cleaning
print(df.info())
print('Data before cleaning:')
print(df.head())
print('\n')
print(df.isna().sum())
print('\n')

# Proportion of male and female
print(df['Gender'].value_counts())
print('\n')


### Data cleaning
# Remove duplicate data and unnecessary columns
df.drop_duplicates()
df.drop(columns=['Transaction ID', 'Customer ID'], inplace=True)

# Remove 2 rows that consisted of 2024 data
print(df.sort_values(by='Date').tail(2))
print('\n')
df.drop([649,210], axis=0, inplace=True)

# Check data accuracy of the total amount
df['Amount'] = df['Quantity'] * df['Price per Unit']
differences = (df['Total Amount'] != df['Amount']).sum()
print(f'Differences: {differences}')
print('\n')

# Rename 'Total Amount' to 'Total Sales'
df.rename(columns={'Total Amount': 'Total Sales'}, inplace=True)


### Data after cleaning
df.drop(columns='Amount', axis=1, inplace=True)
print('Data after cleaning:')
print(df.head())
print('\n')


### 1. How does customer age and gender influence their purchasing behavior?
## 1(a) Relationship between age,  total sales and quantity purchased
corr_age_sales = df['Age'].corr(df['Total Sales'])
corr_age_qty = df['Age'].corr(df['Quantity'])
correlation_age = pd.DataFrame({
    'Correlation': [corr_age_sales,corr_age_qty]
    },index= [['Age vs Sales', 'Age vs Qty']])
print(correlation_age)
print('\n')

## 1(b) Purchasing behavior of different age groups based on total sales
age = df[['Age', 'Product Category', 'Total Sales']].copy()
bins = [18,27,37,47,57,64]
labels = ['18-27', '28-37', '38-47', '48-57', '58-64']
age['Age Group'] = pd.cut(age['Age'], bins=bins, labels=labels, right=True)

# Group 'age' by 'Age Group' and 'Product Category'
grouped_age = age.groupby(['Age Group','Product Category'], observed=False)['Total Sales'].mean().unstack().astype(int)
print(grouped_age)
print('\n')

# Plot for 'grouped_age'
grouped_age.plot(kind='bar', stacked=True)
plt.title('Average Sales by Age Groups and Product Category')
plt.xlabel('Age Groups')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.legend(title='Product Category')
plt.tight_layout()
plt.show(block=False)

## 1(c) Purchasing behavior and preferences on different genders based on total sales
new_df = df[['Gender','Quantity', 'Total Sales','Product Category']]

# Seperate male data out from 'new_df'
male_data = new_df[new_df['Gender'] == 'Male'].copy()
male_data.rename(columns= {'Product Category':'Product Category for Male'}, inplace=True)

# Group 'male_data' by 'Product Category for Male'
male_data = male_data.groupby('Product Category for Male').sum()
male_data.drop('Gender', axis=1, inplace=True)
print(male_data.sort_values(by='Total Sales', ascending=False))
print('\n')

# Seperate female data out from 'new_df'
female_data = new_df[new_df['Gender'] == 'Female'].copy()
female_data.rename(columns={'Product Category':'Product Category for Female'}, inplace=True)

# Group 'female_data' by 'Product Category for Female'
female_data = female_data.groupby('Product Category for Female').sum()
female_data.drop(columns='Gender', axis=1, inplace=True)
print(female_data.sort_values(by='Total Sales', ascending=False))
print('\n')


### 2. Are there discernible patterns in sales across different time periods?
## 2(a) Sales figure in different categories across different months
sales_month = df[['Date', 'Product Category', 'Total Sales']].copy()
sales_month['Date'] = pd.to_datetime(sales_month['Date'])
sales_month['Month'] = sales_month['Date'].dt.month_name()

# Reorder the month into correct order
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
sales_month['Month'] = pd.Categorical(sales_month['Month'], categories=month_order, ordered=True)

# Group 'sales_month' by 'Month' and 'Product Category'
sales_month.drop(columns=['Date'], axis=1, inplace=True)
grouped_sales_month = sales_month.groupby(['Month', 'Product Category'], observed=True).mean().unstack().fillna(0).astype(int)
print(grouped_sales_month.sort_values(by='Month'))
print('\n')

# Plot for 'grouped_sales_month'
grouped_sales_month.plot(kind='bar', stacked=True)
plt.title('Monthly Average Sales by Product Category ')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.legend(df['Product Category'], title='Product Category')
plt.tight_layout()
plt.show(block=False)

## 2(b) Monthly sales
sales_month.drop(columns='Product Category', axis=1, inplace=True)

# Group 'total_sales_month' by 'Month'
total_sales_month = sales_month.groupby('Month', observed=True).sum().sort_values(by='Month')
print(total_sales_month)
print('\n')

# Plot for 'total_sales_month'
total_sales_month.plot.barh(stacked=False)
plt.title('Total Monthly Sales')
plt.xlabel('Total Sales')
plt.ylabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=False)

## 2(c) Total sales across different seasons
sales_season = sales_month.copy()
sales_season['Product Category'] = df['Product Category']
sales_season.rename(columns={'Month':'Season'}, inplace=True)

# Assign month into different seasons
month_season = {
    'December':'Winter', 'January':'Winter', 'February':'Winter',
    'March':'Spring', 'April':'Spring', 'May':'Spring',
    'June':'Summer', 'July':'Summer', 'August':'Summer',
    'September':'Autumn', 'October':'Autumn', 'November':'Autumn',
}
sales_season['Season'] = sales_season['Season'].replace(month_season)

# Reorder season
season_order = ['Spring','Summer','Autumn','Winter']
sales_season['Season'] = pd.Categorical(sales_season['Season'], categories=season_order, ordered=True)

# Group 'sales_season' by 'Season' and 'Product Category'
sales_season = sales_season.groupby(['Season', 'Product Category'], observed=True).mean().unstack().astype(int)
print(sales_season)
print('\n')

# Plot for 'sales_season'
sales_season.plot(kind='bar', stacked=True)
plt.title('Seasonal Average Sales by Product Category')
plt.xlabel('Season')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.legend(df['Product Category'], title='Product Category')
plt.tight_layout()
plt.show(block=False)


### 3. Which product categories hold the highest appeal among customers?
# Total spending amount across different product categories
best_category = df[['Product Category', 'Total Sales']].copy()

# Group 'best_category' by 'Product Category'
best_category = best_category.groupby('Product Category').sum()
print(best_category.sort_values(by='Total Sales', ascending=False))
print('\n')


### 4. What insights can be gleaned from the distribution of product prices within each category?
## 4(a) Average price across different categories
price = df[['Product Category', 'Price per Unit', 'Total Sales']].copy()

# Group 'avg_price' by 'Product Category'
avg_price = price.groupby('Product Category').mean().round().astype(int)
print(avg_price)
print('\n')

## 4(b) Price distribution within each category
price_distribution = price.groupby('Product Category')['Price per Unit'].value_counts().unstack()
print(price_distribution)
print('\n')

### 5. What is the correlation between price, sales and quantity?
corr_price_sales = df['Price per Unit'].corr(df['Total Sales'])
corr_price_quantity = df['Price per Unit'].corr(df['Quantity'])
correlation_price = pd.DataFrame({
    'Correlation': [corr_price_sales,corr_price_quantity]
    },index= [['Price vs Sales', 'Price vs Qty']])
print(correlation_price)

