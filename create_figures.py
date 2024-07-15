#!/usr/bin/env python
# coding: utf-8


# load libraries

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter



# designate the file location

file_location = "/Users/"
os.chdir(file_location)


# open the file containing the news articles

df = pd.read_csv("antisemitism_full_text.csv")


# preprocess the text

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords_dict]
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

df['text'] = df['text'].apply(preprocess_text)


# drop the possible duplicates

df["text_len"] = df["text"].str.len()
df = df.sort_values(by="text_len", ascending=False)
df["first100"] = df["text"].str[:100]
df = df.drop_duplicates(subset=["first100"])


# make the lists for each category

politics = ["biden", "trump", "election", "democrat", "republican", "campaign", "partisan"]
threats = ["nazi", "synagogue", "threat", "violence", "crime", "attack", "slur", "hateful"]
universities = ["student", "university", "universities", "campus", "college", "university president", "higher education"]
zionism = ["zionism"]
genocide = ["genocide"]
charged = ["zionism", "genocide"]


# check whether each article contains the target words or not

df["thr"] = 0 # threats
df["pol"] = 0 # politics
df["uni"] = 0 # universities
df["zion"] = 0 # zionism
df["geno"] = 0 # genocide
df["chrg"] = 0 # charged words: zionism and genocide

for tw in threats:
    df.loc[df["text"].str.contains(tw), "thr"] = 1

for pw in politics:
    df.loc[df["text"].str.contains(pw), "pol"] = 1

for uw in universities:
    df.loc[df["text"].str.contains(uw), "uni"] = 1
    
for zw in zionism:
    df.loc[df["text"].str.contains(zw), "zion"] = 1

for gw in genocide:
    df.loc[df["text"].str.contains(gw), "geno"] = 1

for cw in charged:
    df.loc[df["text"].str.contains(cw), "chrg"] = 1


# divide the time period into three

df["stage"] = ""
df.loc[df["PubDate"] < "2023-10-07", "stage"] = "early"
df.loc[(df["PubDate"] >= "2023-10-07") & (df["PubDate"] < "2024-04-01"), "stage"] = "middle"
df.loc[df["PubDate"] >= "2024-04-01", "stage"] = "late"


# create the numbers in table 2

#df.groupby(["stage"])["ID"].count()
#df.loc[(df["zion"] == 1) & (df["thr"] == 1)].groupby("stage").count()



# create Figure 1

df['date'] = pd.to_datetime(df['PubDate'])
df = df.sort_values('date')
plt.figure(figsize=(15, 8))

df_grouped = df.groupby(pd.Grouper(key='date', freq='W'))['ID'].count().reset_index()
df_grouped["uni"] = df.groupby(pd.Grouper(key='date', freq='W'))['uni'].sum().tolist()
df_grouped["pol"] = df.groupby(pd.Grouper(key='date', freq='W'))['pol'].sum().tolist()
df_grouped["thr"] = df.groupby(pd.Grouper(key='date', freq='W'))['thr'].sum().tolist()
df_grouped["chrg"] = df.groupby(pd.Grouper(key='date', freq='W'))['chrg'].sum().tolist()

plt.plot(df_grouped['date'], df_grouped['ID'], marker='.', linestyle='-', color='black', label="All")
plt.plot(df_grouped['date'], df_grouped['uni'], marker='.', linestyle='-', color='blue', label="Universities")
plt.plot(df_grouped['date'], df_grouped['pol'], marker='.', linestyle='-', color='green', label="Politics")
plt.plot(df_grouped['date'], df_grouped['thr'], marker='.', linestyle='-', color="red", label="Threats")
plt.plot(df_grouped['date'], df_grouped['chrg'], marker='.', linestyle='-', color="orange", label="Threats")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

special_date = pd.to_datetime('2023-10-07')
plt.axvline(x=special_date, color='black', linestyle='-', linewidth=1)
special_date = pd.to_datetime('2024-04-01')
plt.axvline(x=special_date, color='black', linestyle='-', linewidth=1)
special_date = pd.to_datetime('2022-10-16')
plt.axvline(x=special_date, color='black', linestyle='-', linewidth=1)

plt.xlabel('Date (Weekly)')
plt.ylabel('Number of Publications by Month')
plt.legend(frameon=False)



# create figure 2

weekly_correlations = []
for week_start, week_data in df.groupby(pd.Grouper(key='date', freq='W')):
    if len(week_data) > 1:
        comention = week_data.loc[(week_data['uni'] * week_data['chrg']) == 1]["ID"].count()
        weekly_correlations.append({
            'Date-Month': week_start,
            'Universities Only':  (week_data['uni'].sum() - comention) / week_data["ID"].count(),
            'Zion-geno Only':  (week_data['chrg'].sum() - comention) / week_data["ID"].count(),
            'Co-mention':  comention / week_data["ID"].count()
        })

weekly_correlations_df = pd.DataFrame(weekly_correlations)
weekly_correlations_df.set_index('Date-Month', inplace=True)

colors = ['#ade3f0', '#fa9120', '#a8f277']
ax = weekly_correlations_df.plot(kind='area', stacked=True, figsize=(15, 8), color=colors)

special_date = pd.to_datetime('2023-10-07')
plt.axvline(x=special_date, color='black', linestyle='-', linewidth=1)
special_date = pd.to_datetime('2024-04-01')
plt.axvline(x=special_date, color='black', linestyle='-', linewidth=1)
special_date = pd.to_datetime('2022-10-16')
plt.axvline(x=special_date, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Monthly')
plt.ylim(0.2, 1)
ax.set_ylabel('Mention')
ax.set_title('Composition/Co-mention')
ax.legend()
plt.tight_layout()
plt.show()



# create Figure S1

for uni in universities:
    df[uni] = 0

for uni in universities:
    df.loc[df["text"].str.contains(uni), uni] = 1

plt.figure(figsize=(15, 8))

df_grouped = df.groupby(pd.Grouper(key='date', freq='W'))['ID'].count().reset_index()
df_grouped["uni"] = df.groupby(pd.Grouper(key='date', freq='W'))['uni'].sum().tolist()

for uni in universities:
    df_grouped[thr] = df.groupby(pd.Grouper(key='date', freq='W'))[thr].sum().tolist()

plt.plot(df_grouped['date'], df_grouped['uni'], marker='.', linestyle='-', color='black', label="Universities-all")
for uni in universities:
    plt.plot(df_grouped['date'], df_grouped[uni], marker='.', linestyle='-', label=uni)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

special_date = pd.to_datetime('2023-10-07')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)
special_date = pd.to_datetime('2024-04-01')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)
special_date = pd.to_datetime('2022-10-16')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)

plt.xlabel('Date (Weekly)')
plt.ylabel('Publications')
plt.legend(frameon=False)



# create figure S2

for pol in politics:
    df[pol] = 0

for pol in politics:
    df.loc[df["text"].str.contains(pol), pol] = 1

plt.figure(figsize=(15, 8))

df_grouped = df.groupby(pd.Grouper(key='date', freq='W'))['ID'].count().reset_index()
df_grouped["pol"] = df.groupby(pd.Grouper(key='date', freq='W'))['pol'].sum().tolist()

for pol in politics:
    df_grouped[pol] = df.groupby(pd.Grouper(key='date', freq='W'))[pol].sum().tolist()

plt.plot(df_grouped['date'], df_grouped['pol'], marker='.', linestyle='-', color='black', label="Politics-all")
for pol in politics:
    plt.plot(df_grouped['date'], df_grouped[pol], marker='.', linestyle='-', label=pol)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

special_date = pd.to_datetime('2023-10-07')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)
special_date = pd.to_datetime('2024-04-01')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)
special_date = pd.to_datetime('2022-10-16')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)

plt.xlabel('Date (Weekly)')
plt.ylabel('Publications')
plt.legend(frameon=False)


# create figure S3

df['date'] = pd.to_datetime(df['PubDate'])
df = df.sort_values('date')

for thr in threats:
    df[thr] = 0

for thr in threats:
    df.loc[df["text"].str.contains(thr), thr] = 1

plt.figure(figsize=(15, 8))

df_grouped = df.groupby(pd.Grouper(key='date', freq='W'))['ID'].count().reset_index()
df_grouped["thr"] = df.groupby(pd.Grouper(key='date', freq='W'))['thr'].sum().tolist()

for thr in threats:
    df_grouped[thr] = df.groupby(pd.Grouper(key='date', freq='W'))[thr].sum().tolist()

plt.plot(df_grouped['date'], df_grouped['thr'], marker='.', linestyle='-', color='black', label="Threats-all")
for thr in threats:
    plt.plot(df_grouped['date'], df_grouped[thr], marker='.', linestyle='-', label=thr)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

special_date = pd.to_datetime('2023-10-07')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)
special_date = pd.to_datetime('2024-04-01')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)
special_date = pd.to_datetime('2022-10-16')
plt.axvline(x=special_date, color='black', linestyle='--', linewidth=1)

plt.xlabel('Date (Weekly)')
plt.ylabel('Publications')
plt.legend(frameon=False)




