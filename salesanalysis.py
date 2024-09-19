## 1. How does customer age and gender influence their purchasing behavior?
## 2. Are there discernible patterns in sales across different time periods?
## 3. Which product categories hold the highest appeal among customers?
## 4. What insights can be gleaned from the distribution of product prices within each category?


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('retailsales.csv')

# Data before cleaning
print(df)
print(df.isna().sum())
print('\n')

# Proportion of male and female
print(df['Gender'].value_counts())
print('\n')


## Data cleaning
# Remove duplicate data and unnecessary columns
df.drop_duplicates()
df.drop(columns=['Transaction ID', 'Customer ID'], inplace=True)

# Remove 2 rows that consisted 2024 data
df.drop([649,210], axis=0, inplace=True)

# Check data accuracy of the total amount
df['Amount'] = df['Quantity'] * df['Price per Unit']
differences = (df['Total Amount'] != df['Amount']).sum()
print(f'Differences: {differences}')
print('\n')


# Data after cleaning
df.drop(columns='Amount', axis=1, inplace=True)
print(df)
print('\n')


## 1. How does customer age and gender influence their purchasing behavior?
# 1(a) Relationship between age, total amount spend and quantity purchased
corr_age_amount = df['Age'].corr(df['Total Amount'])
corr_age_qty = df['Age'].corr(df['Quantity'])
correlation_age = pd.DataFrame({
    'Correlation': [corr_age_amount,corr_age_qty]
    },index= [['Age vs Amount', 'Age vs Qty']])
print(correlation_age)
print('\n')

# 1(b) Purchasing behavior of different age groups based on total amount spent
age = df[['Age', 'Product Category', 'Total Amount']].copy()
bins = [18,27,37,47,57,64]
labels = ['18-27', '28-37', '38-47', '48-57', '58-64']
age['Age Group'] = pd.cut(age['Age'], bins=bins, labels=labels, right=True)
grouped_age = age.groupby(['Age Group','Product Category'], observed=False)['Total Amount'].sum().unstack()
print(grouped_age)
print('\n')

# Plot for 1(b)
grouped_age.plot(kind='bar', stacked=True)
plt.title('Total Amount by Age Groups and Product Category')
plt.xlabel('Age Groups')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
plt.legend(title='Product Category')
plt.tight_layout()
plt.show()

# 1(c) Purchasing behavior and preferences on different genders based on total amount spent
new_df = df[['Gender','Quantity', 'Total Amount','Product Category']]

male_data = new_df[new_df['Gender'] == 'Male'].copy()
male_data.rename(columns= {'Product Category':'Product Category for Male'}, inplace=True)
male_data = male_data.groupby('Product Category for Male').sum()
male_data.drop('Gender', axis=1, inplace=True)
print(male_data.sort_values(by='Total Amount', ascending=False))
print('\n')

female_data = new_df[new_df['Gender'] == 'Female'].copy()
female_data.rename(columns={'Product Category':'Product Category for Female'}, inplace=True)
female_data = female_data.groupby('Product Category for Female').sum()
female_data.drop(columns='Gender', axis=1, inplace=True)
print(female_data.sort_values(by='Total Amount', ascending=False))
print('\n')


## 2. Are there discernible patterns in sales across different time periods?
# 2(a) Sales figure in different categories across different months
sales_month = df[['Date', 'Product Category', 'Total Amount']].copy()
sales_month['Date'] = pd.to_datetime(sales_month['Date'])
sales_month['Month'] = sales_month['Date'].dt.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
sales_month['Month'] = pd.Categorical(sales_month['Month'], categories=month_order, ordered=True)
sales_month.drop(columns=['Date'], axis=1, inplace=True)
grouped_sales_month = sales_month.groupby(['Month', 'Product Category'], observed=True).sum().unstack().fillna(0).astype(int)
print(grouped_sales_month.sort_values(by='Month'))
print('\n')

# Plot for 2(a)
grouped_sales_month.plot(kind='bar', stacked=True)
plt.title('Monthly Sales by Product Category')
plt.xlabel('Month')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
my_labels = ['Beauty', 'Clothing', 'Electronics']
plt.legend(title='Product Category', labels=my_labels)
plt.tight_layout()
plt.show()


# 2(b) Monthly sales
sales_month.drop(columns='Product Category', axis=1, inplace=True)
total_sales_month = sales_month.groupby('Month', observed=True).sum().sort_values(by='Month')
print(total_sales_month)
print('\n')

# Plot for 2(b)
total_sales_month.plot(kind='bar', stacked=False, legend=False)
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2(c) Total sales across different seasons
sales_season = sales_month.copy()
sales_season['Product Category'] = df['Product Category']
month_season = {
    'December':'Winter', 'January':'Winter', 'February':'Winter',
    'March':'Spring', 'April':'Spring', 'May':'Spring',
    'June':'Summer', 'July':'Summer', 'August':'Summer',
    'September':'Autumn', 'October':'Autumn', 'November':'Autumn',
}
sales_season['Month'] = sales_season['Month'].replace(month_season)
sales_season.rename(columns={'Month':'Season'}, inplace=True)
sales_season = sales_season.groupby(['Season', 'Product Category'], observed=True).sum().unstack()
print(sales_season)
print('\n')


## 3. Which product categories hold the highest appeal among customers?
# Total spending amount across different product categories
best_product = df[['Product Category', 'Total Amount']].copy()
best_product = best_product.groupby('Product Category').sum()
print(best_product.sort_values(by='Total Amount', ascending=False))
print('\n')


## 4. What insights can be gleaned from the distribution of product prices within each category?
# 4(a) Average price across different categories
price = df[['Product Category', 'Price per Unit', 'Total Amount']].copy()
avg_price = price.groupby('Product Category').mean().round().astype(int)
print(avg_price)
print('\n')

# 4(b) Price distribution within each category
price_distribution = price.groupby('Product Category')['Price per Unit'].value_counts().unstack()
print(price_distribution)