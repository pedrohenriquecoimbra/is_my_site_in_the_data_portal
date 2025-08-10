import requests
import re
from io import StringIO

import numpy as np
import pandas as pd
import calendar
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
sns.set_theme(context='notebook', style='white')


def get_data_from_url(url=None, *args, **kwargs):

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        in_memory = StringIO(response.content.decode('utf-8'))
        data = pd.read_csv(in_memory, *args, **kwargs)
    else:
        print(f"Failed to download file: {response.status_code}")
        data = pd.DataFrame()
    return data


def findall_externalUploadsLog():
    return ['https://data.icos-cp.eu/upload/etc/staging/' + p for p in re.findall("(externalUploadsLog.txt.*)\"",
                                                                                  requests.get('https://data.icos-cp.eu/upload/etc/staging/').content.decode('utf-8'))]


def data_from_externalUploadsLog():
    # errorLog, externalUploadsLog
    df = pd.concat([get_data_from_url(p, sep='\t', names=[
                   'TIMESTAMP', 'filename']) for p in findall_externalUploadsLog()])
    # get_data_from_url('https://data.icos-cp.eu/upload/etc/staging/externalUploadsLog.txt', sep='\t', names=['TIMESTAMP', 'filename'])
    # get_data_from_url('https://data.icos-cp.eu/upload/etc/staging/errorLog.txt', sep='\t', names=['TIMESTAMP', 'FILENAME'])

    # remove duplicate values (e.g. file submitted multiple times)
    df.sort_values(by='filename', inplace=True)
    df.drop_duplicates(subset=['filename'], keep='last', inplace=True)

    # split file name into components, e.g. FR-Hes_EC_202106252130_L10_F01.zip
    df = pd.concat(
        [df['TIMESTAMP'], df['filename'].str.split('_', expand=True)], axis=1)
    df.rename(columns={0: 'site', 1: 'filetype', 2: 'date',
                       3: 'logger', 4: 'endfile'}, inplace=True)

    # Split EC half-hourly files from daily files
    hh = df.copy()  # df[df['filetype'] == 'EC'].copy()
    # pp = df[df['filetype'] == 'PHEN'].copy()
    # dd = df[(df['filetype'] != 'EC') & (df['filetype'] != 'PHEN')].copy()
    hh['datetime'] = pd.to_datetime(
        hh['date'].str.ljust(12, '0'), format='%Y%m%d%H%M')
    # pp['datetime'] = pd.to_datetime(pp['date'], format='%Y%m%d%H%M')
    # dd['datetime'] = pd.to_datetime(dd['date'], format='%Y%m%d')
    hh['datetime_ym'] = pd.to_datetime(hh['date'].str[:6], format='%Y%m')
    del df
    return hh


def datetime_missing(hh, dt_round='1M', index_on=['site'], expected_per_day=48):
    # Round always uppercase
    dt_round = dt_round.upper()

    # Round datetime to monthly/daily/hourly/...
    if dt_round in ['1M', 'M']:
        hh['datetime'] = hh['datetime_ym']
    elif dt_round:
        hh['datetime'] = hh['datetime'].dt.floor(dt_round)

    # EC logs are available between:
    dstart = hh['datetime'].min()
    dend = hh['datetime'].max()
    print(f'Start date: {dstart}')
    print(f'End date:   {dend}')

    hh['expected_per_day'] = np.where(hh['date'].str.len() == 8, 1, 48)
    # Missing EC files per month and year
    # available files
    # yis = hh.groupby(['site', 'datetime'])['site'].count().rename('value')
    yis = hh.groupby(['site', 'logger', 'endfile', 'datetime'])['site'].count(
    ).rename('value').reset_index().groupby(index_on + ['datetime'])['value'].min()
    
    # yis = yis.reset_index().groupby(['site', 'datetime'])['value'].min()
    # should be 48 files per day
    if dt_round in ['1M', 'M']:
        ymax = (hh['datetime'].dt.daysinmonth * hh['expected_per_day']
                ).groupby([hh[i] for i in index_on] + [hh['datetime']]).min()
    elif dt_round == '1D':
        ymax = hh.groupby(index_on + ['datetime'])['expected_per_day'].max()
    elif dt_round == '1h':
        ymax = hh.groupby(index_on + ['datetime']
                          )['expected_per_day'].max() / 24
    elif not dt_round:
        ymax = 1
    else:
        ymax = np.nan
    # missing files
    ymiss = ymax - yis
    # pandas.Series with mulitindex
    ymiss.index.names = index_on + ['datetime']
    # mulitindex to column
    ymiss = ymiss.reset_index()
    # make 2d array
    ymiss = ymiss.pivot(index=index_on, columns='datetime')
    # reorder from recent to old
    ymiss = ymiss.T.sort_index(ascending=False).T

    ymiss.columns = [c[1] for c in ymiss.columns]
    ymiss.columns.names = ['datetime']
    return ymiss


def plot_asLafontCuntz(hh, sitename='FR-Hes', dst_fig=None):
    hh = hh[hh['filetype'] == 'EC'].copy()
    if sitename:
        hh = hh[hh['site'] == sitename].copy()

    # EC logs are available between:
    dstart = hh['datetime'].min()
    dend = hh['datetime'].max()
    print(f'Start date: {dstart}')
    print(f'End date:   {dend}')

    # Missing EC files per month and year
    # available files
    yis = hh['site'].groupby(
        [hh['datetime'].dt.month, hh['datetime'].dt.year]).count()
    # should be 48 files per day
    ymax = hh['datetime'].dt.daysinmonth.groupby(
        [hh['datetime'].dt.month, hh['datetime'].dt.year]).min() * 48
    # missing files
    ymiss = ymax - yis
    # pandas.Series with mulitindex
    ymiss.index.names = ['month', 'year']
    # mulitindex to column
    ymiss = ymiss.reset_index()
    # make 2d array
    ymiss = ymiss.pivot(index='month', columns='year')

    ymin = hh['datetime'].dt.year.min()
    ymax = hh['datetime'].dt.year.max()
    ax = sns.heatmap(ymiss, vmax=31*48, annot=True, fmt=".0f", linewidths=0.5,
                     cmap='Oranges',
                     xticklabels=np.arange(ymin, ymax + 1))
    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    if dst_fig:
        plt.savefig(dst_fig.rsplit('.', 1)[
                    0] + '_MM_asLafontCuntz.png', bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # Missing EC files per hour and day for the last two months
    today = dt.datetime.today()
    if today.month == dend.month:
        isend = today
    else:
        isend = dend
    month_before = isend - dt.timedelta(days=isend.day + 1)
    months = [month_before, isend]

    for imonth in months:
        mm = hh[(hh['datetime'].dt.year == imonth.year) &
                (hh['datetime'].dt.month == imonth.month)]
        yis = mm['site'].groupby(
            [mm['datetime'].dt.hour, mm['datetime'].dt.day]).count()
        ymax = 2  # two half hourly files per hour
        ymiss = ymax - yis
        ymiss.index.names = ['hour', 'day']
        ymiss = ymiss.reset_index()
        ymiss = ymiss.pivot(index='hour', columns='day')

        days = calendar.monthrange(imonth.year, imonth.month)[
            1]  # days in month
        ax = sns.heatmap(ymiss, vmax=2, annot=True, fmt=".0f", linewidths=0.5,
                         cmap='Oranges', xticklabels=np.arange(days + 1))
        ax.set_xlabel('Day')
        ax.set_ylabel('Hour')
        if dst_fig:
            plt.savefig(dst_fig.rsplit('.', 1)[
                        0] + f'_{imonth.strftime("%m")}{imonth.year}_asLafontCuntz.png', bbox_inches='tight')
            if imonth == month_before:
                plt.savefig(dst_fig.rsplit('.', 1)[
                            0] + f'_lastmonth_asLafontCuntz.png', bbox_inches='tight')
            if imonth == isend:
                plt.savefig(dst_fig.rsplit('.', 1)[
                            0] + f'_currmonth_asLafontCuntz.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def plot_allsites(ymiss, sitename='FR-'):
    if sitename:
        ymiss = ymiss[ymiss.index.map(lambda x: sitename in x)].copy()

    plt.figure(figsize=(50, 6))
    ax = sns.heatmap(ymiss, vmax=31*48, annot=True, fmt=".0f", linewidths=0.5,
                     cmap='Oranges',
                     )
    plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%Y'))
    plt.show()


def plot_allsites_i(ymiss, dst_fig=None, **kwargs):
    import plotly.express as px
    import plotly.io as pio
    import numpy as np
    np.bool = np.bool_
    # save columns, reset_index, filter sitename, set_index cols-saved_columns
    # if sitename: ymiss = ymiss[ymiss.index.map(lambda x: sitename in x)].copy()
    custom_cmap = [[c[0]/2, c[1]] for c in px.colors.get_colorscale('Blues_r')[:-1]] + [
        [0.5+c[0]/2, c[1]] for c in px.colors.get_colorscale('Oranges')[:-1] + px.colors.get_colorscale('Oranges')]
    # Range color with 0 in the middle. Saturate at max (and not min) to ensure 0 in the middle.
    custom_range = [min(-abs(np.max(np.max(ymiss))), -1),
                    max(abs(np.max(np.max(ymiss))), 1)]
    
    fig = px.imshow(ymiss, text_auto=True, aspect="auto", labels={'y': '', 'x': ''},
                    color_continuous_scale=px.colors.sequential.RdBu_r,
                    range_color=custom_range, color_continuous_midpoint=0)
    fig.update_layout(  # xaxis = dict(range=range, maxallowed=max(ymiss.columns)+pd.Timedelta('15D'), minallowed=min(ymiss.columns), fixedrange=False),
        yaxis=dict(maxallowed=len(ymiss)+.5, minallowed=-.5,
                   dtick='M1', fixedrange=True),
        coloraxis_colorbar=dict(thickness=20),
        # width=800,
        # height=1000,
        dragmode='pan',  # Enable only panning
        # title='Missing data:',
        margin=dict(l=20, r=20, t=20, b=20, pad=0),
        **kwargs)
    if dst_fig:
        pio.write_html(fig, file=dst_fig, full_html=True,
                       config={'scrollZoom': True}, default_height=30*ymiss.shape[0])
    else:
        fig.show()


def main(site_name=[None],
         date_range=['1M', '1D', '1h'],
         dst_figure='fr-icos-paul/assets/snippets/ETC_{}{}_missing_uploads.html',
         figure_type=['i'],
         index_on=['site']):
    df = data_from_externalUploadsLog()
    xaxis = dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date")

    for rg in date_range:
        print(dst_figure, rg, figure_type)
        for s in site_name:
            print(s)
            try:
                dst_figure_full = dst_figure.format("" if s == None else s, rg)
                if 's' in figure_type:
                    plot_asLafontCuntz(df, s, dst_figure_full)

                if 'i' in figure_type:
                    df_ = df.copy().query(
                        f'site.astype("str").str.contains("{s}")') if s else df.copy()
                    if not rg:
                        df_ = df_.head(48 * 180)
                    if len(index_on) <= 1:
                        df_ = df_[df_['filetype'] == "EC"]
                    ymiss = datetime_missing(
                        df_, rg, index_on=index_on).copy()
                    if len(index_on) > 1:
                        ymiss = ymiss.set_axis(['_'.join(list(el)) for el in list(zip(
                            *[ymiss.index.get_level_values(i).astype(str).values for i in range(len(index_on))]))], axis='index')
                    
                    if 'M' in rg.upper():
                        _range = (
                            max(ymiss.columns[18:-18]), max(ymiss.columns)+pd.Timedelta('15D'))
                        maxallowed = max(ymiss.columns)+pd.Timedelta('15D')
                    elif 'D' in rg.upper():
                        _range = (
                            max(ymiss.columns[46:-46]), max(ymiss.columns)+pd.Timedelta('12h'))
                        maxallowed = max(ymiss.columns)+pd.Timedelta('12h')
                    elif 'H' in rg.upper():
                        _range = (
                            max(ymiss.columns[80:-80]), max(ymiss.columns)+pd.Timedelta('30Min'))
                        maxallowed = max(ymiss.columns)+pd.Timedelta('30Min')
                    else:
                        _range = (
                            max(ymiss.columns[80:-80]), max(ymiss.columns)+pd.Timedelta('15Min'))
                        maxallowed = max(ymiss.columns)+pd.Timedelta('15Min')
                    xaxis.update(dict(range=_range, maxallowed=maxallowed, minallowed=min(
                        ymiss.columns), fixedrange=False))

                    plot_allsites_i(ymiss, dst_fig=dst_figure_full, xaxis=xaxis)
            except Exception as e:
                print(f'err: {str(e)}')
                continue
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--site_name', type=str,
                        nargs='+', default=[None])
    parser.add_argument('-d', '--date_range', type=str,
                        nargs='+', default=['1M', '1D', '1h'])
    parser.add_argument('-out', '--dst_figure', type=str,
                        default='fr-icos-paul/assets/snippets/ETC_{}{}_missing_uploads.html')
    parser.add_argument('-t', '--figure_type', type=str,
                        nargs='+', default=['i'])
    parser.add_argument('-i', '--index_on', type=str,
                        nargs='+', default=['site'])
    args = parser.parse_args()
    argd = vars(args)
    main(**argd)
