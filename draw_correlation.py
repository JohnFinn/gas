#!/usr/bin/env python3
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import itertools


def read_data():
    df : pd.DataFrame = pd.read_excel('Export_GTF_IEA.XLS')
    df = df.replace('#N/A()', None)
    df = df[:195]
    df = df[(df['Exit'] != 'Liquefied Natural Gas') & (df['Entry'] != 'Liquefied Natural Gas')]
    return df

df = read_data()
flow_by_month = df[[c for c in df.columns if isinstance(c, dt.datetime)]].astype(float).dropna()


def draw_correlation(data: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    mshow0 = axes[0].matshow(data.corr(), vmin=0, vmax=1)
    mshow1 = axes[1].matshow(data.T.corr(), vmin=0, vmax=1)
    axes[0].set_xticklabels(data.columns)
    axes[0].set_yticklabels(data.columns)

    cb0 = fig.colorbar(mshow0)
    plt.gcf().set_size_inches(50, 30)
    # plt.xticks(data.columns, rotation=45)
    plt.savefig('correlation.png')
    plt.show()

def draw_all(data: pd.DataFrame, labels):
    for (idx, row), label in zip(data.iterrows(), labels):
        plt.plot(row.index, row, label=label)
    plt.xticks(row.index, rotation=45)
    plt.legend()
    plt.gcf().set_size_inches(100, 70)
    plt.savefig('flow.png')
    plt.show()

# add months to x axis
draw_correlation(flow_by_month)
# draw_all(flow_by_month[:15], (df['Exit'] + ' -> ' + df['Entry'])[:15])
