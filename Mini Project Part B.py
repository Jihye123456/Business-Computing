# -*- coding: utf-8 -*-
"""
22508570 Jihye Lim
22236155 Liu Wing Chi
"""
#%% Q1.

import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import seaborn as sns
import numpy as np

#a.	Produce two visualization (tables or charts) to talk about the yearly salary by different positions,
# and the relationship between years of experience and yearly salary.

# yearly salary by different positions
pd.set_option('display.max_rows', None)
d = pd.read_csv("C:/Users/admin/Desktop/코딩공부/비즈니스 컴퓨팅/IT_Salary.csv")
pos_sal = d.loc[:, ['Yearly salary', 'Position ']]

#count values
pos_types = pos_sal['Position '].value_counts().reset_index(drop=False)
pos_types.reset_index()


# count the position more than 30 ppl in. 
# backend developer, software engineer, data scientist, frontend developer, devops,
# mobile developer, fullstack developer, qa engineer, manager, qa, product manager, ml engineer, software architect


# Define a list of the desired positions
positions = ['backend developer', 'software engineer', 'data scientist', 'frontend developer', 'devops', 
             'mobile developer', 'fullstack developer', 'qa engineer', 'manager', 'qa', 
             'product manager', 'ml engineer', 'software architect', 'machine learning engineer']

# Filter the DataFrame to include only the desired positions
filtered_data = pos_sal[pos_sal['Position '].str.lower().isin(positions)]

# Create a new DataFrame with the median salary for each position
median_salary_by_position = filtered_data.groupby('Position ').median()['Yearly salary'].reset_index()

# Create a bar chart using seaborn
sns.set(rc = {'figure.figsize':(30,8)})

sns.barplot(x='Position ', y='Yearly salary', data=median_salary_by_position).set(title='Salary by Positions')


# relationship between years of experience and yearly salary
year_sal = d.loc[:, ['Yearly salary', 'Total years of experience']]

#separated into different experience categories

year_range = [0,3,6,9,12,15,float('inf')]
year_labels = ['0-3y', '3-6y', '6-9y', '9-12y', '12-15y', '15yk+']
year_sal['Year Category']= pd.cut(year_sal['Total years of experience'], bins=year_range, labels=year_labels)


# Create a new DataFrame with the median salary for each year category
median_salary_by_year = year_sal.groupby('Year Category').median()['Yearly salary'].reset_index()

# Create a bar chart using matplotlib

fig, ax = plt.subplots(figsize=(12,8))

# Define custom colors
colors = ['mistyrose', 'lightcoral', 'indianred', 'brown', 'firebrick', 'maroon']

# Create the bar plot
ax.bar(median_salary_by_year['Year Category'], median_salary_by_year['Yearly salary'], color = colors)

# Set the x-axis label and y-axis label
ax.set_xlabel('Year Category')
ax.set_ylabel('Yearly salary')
ax.set_title('Salary by Years of experience')


#b

# removing outlier - Berlin
    
berlin_samp = d[d['City'] == 'Berlin']['Yearly salary']

ber_level_1q = berlin_samp.quantile(0.25)
ber_level_3q = berlin_samp.quantile(0.75)

IQR = ber_level_3q - ber_level_1q

remove_range = 1.5 #removing range


berlin= berlin_samp[(berlin_samp <= ber_level_3q + (remove_range * IQR)) 
          & (berlin_samp >= ber_level_1q - (remove_range*IQR))]

# removing outlier - Other cities

oth_samp = d[d['City'] != 'Berlin']['Yearly salary']

oth_level_1q = oth_samp.quantile(0.25)
oth_level_3q = oth_samp.quantile(0.75)


IQR = oth_level_3q - oth_level_1q

remove_range = 1.5 #removing range

others = oth_samp[(oth_samp <= oth_level_3q + (remove_range * IQR)) 
          & (oth_samp >= oth_level_1q - (remove_range*IQR))]



# find out p_value and make plot
mean_berlin = berlin.mean()
sd_berlin = berlin.std()
mean_others = others.mean()
n = len(berlin)
test_stat = (mean_berlin - mean_others)/sd_berlin * math.sqrt(n)
print('test statistic: ', test_stat)
p_value = (1-stats.t.cdf(test_stat,df = n-1))
print('p-value: ', p_value)

data = [berlin, others]

fig,ax = plt.subplots(figsize = (8, 10))

ax.boxplot(data)
ax.set_title('Yearly salary by cities')
ax.set_xticklabels(['Berlin','Other cities'])
plt.show()

#test statistic:  6.461825921022834
# p-value:  7.886091779596427e-11
# conclusion: since p value is smaller than alpha(0.05), we can reject null hypothesis.
# There is sufficient evidence to show that the average IT salary in Berlin is higher than that in other cities. 



#c. 
#In terms of yearly salary by position, software architect has highest salary, and manager, machine learning engineer, devops, software engineer, backend developer, product manager, data scientist, fullstack developer, frontend developer, mobile developer, ml engineer, qa, qa engineer. 
#In terms of yearly salary by year of experience, the yearly salary increased as the year of experience increased. 
#In terms of salary in Berlin compared to other cities, since p value is smaller than alpha(0.05), there is sufficient evidence to show that the average IT salary in Berlin is higher than that in other cities. 


#%% Q2.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d =  pd.read_csv("C:/Users/admin/Desktop/코딩공부/비즈니스 컴퓨팅/IT_Salary.csv")

#we need to use only salary and age -> remove other columns
df = d.drop(['City', 'Seniority level', 'Seniority level',
           'Total years of experience', 'Gender',
            'Number of vacation days', 'Main language at work',
            'Company type', 'Contract', 'Year'], axis='columns')


#a. data cleaning - remove outlier


level_1q = df.quantile(0.25)
level_3q = df.quantile(0.75)

IQR = level_3q - level_1q

remove_range = 1.5 #removing range

df_2 = df[(df <= level_3q + (remove_range * IQR)) 
          & (df >= level_1q - (remove_range)*IQR)]

df_sal = df_2.dropna(subset = ['Yearly salary'])
df_age = df_2.dropna(subset = ['Age'])
                   
#b. Verify the central limit theorem using salary data 


df_salary = df_sal['Yearly salary']

sample_sizes = [10, 50, 100, 500, 1000]
for sample_size in sample_sizes:
    averages_sal = []
    for iteration in range(1000):
        samples = np.random.choice(df_salary, size = sample_size)
        sample_mean = samples.mean()
        averages_sal.append(sample_mean)
    plt.figure(figsize=(15,8))
    plt.hist(averages_sal, bins = 100, color = 'C2', histtype = 'stepfilled')
    plt.axvline(x=np.mean(averages_sal), color='C3', linestyle='dashed', linewidth=2)
    plt.title(f'Histogram of Sample Means_salary (n={sample_size}, N=1000)')
    plt.xlabel('Sample Mean_salary')
    plt.ylabel('Frequency')
    plt.show()

    
#c. Verify the central limit theorem using age data

df_age = df_age['Age']

sample_sizes = [10, 50, 100, 500, 1000]
for sample_size in sample_sizes:
    averages_age = []
    for iteration in range(1000):
        samples = np.random.choice(df_age, size = sample_size)
        sample_mean = samples.mean()
        averages_age.append(sample_mean)
    plt.figure(figsize=(15,8))
    plt.hist(averages_age, bins = 100, color = 'C9', histtype = 'stepfilled')
    plt.axvline(x=np.mean(averages_age), color='C3', linestyle='dashed', linewidth=2)
    plt.title(f'Histogram of Sample Means_age (n={sample_size}, N=1000)')
    plt.xlabel('Sample Mean_age')
    plt.ylabel('Frequency')
    plt.show()

    
#d. Discuss your findings

#From results in part b and c, with the use of bootstrapping, randomly choosing the samples in the population by
#columns 'Yearly salary' and 'Age'. With different number of sample sizes, n equals to 10, 50, 100, 500 or 1000, it produces 
#different sample means and using histograms to determine the trends of the data. 

#From the 10 histograms we plotted, it shows that with an increasing sample size, the distribution is tend to be more nearer as 
#a normal distribution. Ther frequencies of data with n = 1000 are most central-focused with an approximate normal distribution.
#In theories, the central limit therom states that the sampling distribution of sample means will turn out as a normal 
#distribution when there is an increasing sample size. Therefore, it could be applied in these two cases based on the data
#supporting the theory

#In addition, central limit theorem mentioned that if fitting the theory, the sample means would be approximately equal to the
#population mean for which the 10 histograms could clearly present the results. In those histograms, a significant change could 
#also be found. There are a change in difference of data in sampling distribution, which means that there is a decreasing trend
#in standard deviation. This has also fit the central limit theorem as it states that the standard deviation of graphs would be
#decreasaed while there is an increasing sample size.
    

#%% Q3

#Find the relationship between seniority level and number of vacation days (>=1). 
#(a. using test of independence) 
#(b. correlation test)

import scipy.stats as stats
import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

#a (test of independence)
#H0: There is no relationship between seniority level and number of vacation days (>=1).
#H1: There is a relationship between seniority level and number of vacation days (>=1).
alpha = 0.05
df = pd.read_csv("C:/Users/admin/Desktop/코딩공부/비즈니스 컴퓨팅/IT_Salary.csv")

#drop outliers
z_scores = np.abs((df['Number of vacation days'] - df['Number of vacation days'].mean())) / df['Number of vacation days'].std()
outlier_row = z_scores[(z_scores > 3)]
df = df.drop(outlier_row.index)

#adding a new column
df['>= 1 days'] = df['Number of vacation days'] >= 1

#make a crosstab table
table = pd.crosstab(df['Seniority level'],df['>= 1 days'])

#conduct the test
stat, p_value, dof, expected = stats.chi2_contingency(table)

print("statistics: ",stat)
print("p-value: ", p_value)
print("degree of freedom: ", dof)
print("expected frequencies: ")
print(expected)

if p_value>alpha:
    print("Fail to reject the null hypothesis")
else:
    print("Reject the null hypothesis")
        
#Conclusion:
#There is no relationship between seniority level and number of vacation days (>=1).

#b (correlation test)

#make a new df 
senlev = df[['Seniority level','Yearly salary', 'Number of vacation days']]

#reset index
senlev = senlev.reset_index(drop=True)
senlev.insert(0, 'Index', range(1, len(senlev)+1))

#calculating the correlation
corr = pg.corr(senlev['Index'],senlev['Number of vacation days'])
print(corr)

#make a scatter plot
plt.scatter(senlev['Index'],senlev['Number of vacation days'])

#Conclusion: From the scatter diagrams, we can determine that there are no relationship between seniority level and 
#            number of vacation days.

#%% Q4

#Determine whether there is a significant difference in the mean yearly salary of different company types at a 0.05 significance level. 
#(a. find which ANOVA test should be used) 
#(b. use the chosen ANOVA test to determine the results) 
#(c. use the correct post-hoc test to determine the differences)
import pingouin as pg

pd.set_option('display.max_columns', None)


sampledf = d.loc[:, ['Yearly salary', 'Company type']]


#a. data cleaning - remove outlier and data preprocessing

level_1q = sampledf['Yearly salary'].quantile(0.25)
level_3q = sampledf['Yearly salary'].quantile(0.75)

IQR = level_3q - level_1q

remove_range = 1.5 #removing range

df_sample_2 = sampledf[(sampledf['Yearly salary'] <= level_3q + (remove_range * IQR)) 
          & (sampledf['Yearly salary'] >= level_1q - (remove_range)*IQR)]

types = df_sample_2['Company type'].value_counts() 
# -> Product, Consulting / Agency, Startup is top 3 frequent company types
#so, anova df should be these 3 to see if there is significant differences between 3 types of companies. 

anova_df = df_sample_2[df_sample_2['Company type'].isin(["Product", "Consulting / Agency", "Startup"]) 
                       & (df_sample_2['Company type'] != "Consulting and Product")]

#%%
#b. test one-way anova

#Test equality of variance
stats = pg.homoscedasticity(anova_df, dv = 'Yearly salary', group = 'Company type')
print(stats)


#               W      pval  equal_var
#levene  0.233932  0.791438       True


# Perform one-way ANOVA

aov = pg.anova(data = anova_df, dv = 'Yearly salary', between = 'Company type')

print(aov)

#         Source  ddof1  ddof2          F         p-unc       np2
#0  Company type      2   1927  20.080183  2.338394e-09  0.020415

#Since the p-value is smaller than the alpha (0.05), so we reject the null hypothesis of the test.
#Conclusion: There is significant difference between company type and yearly salary.

#%%
#c. Tukey post-hoc test 
# differences between company type and yearly salary

tukey = pg.pairwise_tukey(data = anova_df, dv = 'Yearly salary', between = 'Company type')

print(tukey)

#                     A        B       mean(A)       mean(B)         diff  \
#0  Consulting / Agency  Product  63693.413793  70589.725173 -6896.311380   
#1  Consulting / Agency  Startup  63693.413793  68641.002506 -4947.588713   
#2              Product  Startup  70589.725173  68641.002506  1948.722667   

#            se         T       p-tukey    hedges  
#0  1103.144229 -6.251505  1.496853e-09 -0.447731  
#1  1277.843032 -3.871828  3.284498e-04 -0.312230  
#2   885.872233  2.199778  7.145235e-02  0.126323  

#Comparing these differences with the p-value, we can determine that the differences 
# between Consulting/Agency and Product and Consulting / Agency and Startup are significant.
# pvalue = 1.496853e-09 (comparison 0), 3.284498e-04(comparison 2)
