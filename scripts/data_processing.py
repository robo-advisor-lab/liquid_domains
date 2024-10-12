import pandas as pd
import numpy as np
import random
import os
import sys
import requests
import time
import datetime as dt

from dotenv import load_dotenv
from flipside import Flipside
from prophet import Prophet

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

from scripts.utils import flipside_api_results, set_random_seed, is_brandable, min_levenshtein_distance, is_subdomain,entropy,add_domain_rank
from scripts.pull_data import pull_data
from models.forecasters import EnsemblePredictor, Prophet_Domain_Valuator, Domain_Valuator, train_ridge_model, train_randomforest_model, train_prophet_model
# from sql_queries.sql_scripts import three_dns_sales

# %%
pd.options.display.float_format = '{:,.2f}'.format


# %%
current_directory = os.getcwd()
current_directory

# %%
load_dotenv()

def process_data(seed=20,api=False, web2_data = False,tld_weight=dict(com=None,net=None,box=None,org=None,eth=None),temporals=True,corr_type='pearson',
                 threshold=None,correlation_analysis=False, last_dataset=False):

    # %%
    set_random_seed(seed)

    print(f'starting process_data')

    if last_dataset:
        print('using last dataset') 
        combined_dataset = pd.read_csv('../data/last_dataset.csv')
        features = combined_dataset.drop(columns=target).columns
        print(f'target:{target},\n features:{features}')

        if columns_to_drop is None:
            columns_to_drop = []

        columns_to_drop = pd.Index(columns_to_drop)  # Ensure it's array-like
        prophet_columns_to_drop = pd.Index(prophet_columns_to_drop)
        # Drop columns from Index
        gen_features = features.difference(columns_to_drop)
        print(f'gen features {gen_features}')
        gen_features

        # %%
        prophet_features = features.difference(prophet_columns_to_drop)
        print(f'prophet features {prophet_features}')
        prophet_features

        # %%
        X = combined_dataset[gen_features]
        y = combined_dataset[target]

        print(f'X: {X} \ny: {y}')

        web3_data = combined_dataset[combined_dataset['web3']==True]

        X_web3 = web3_data[gen_features]
        y_web3 = web3_data[target]

        print(f'done process_data')

        return X, y, prophet_features, gen_features, target, combined_dataset, features, web3_data, X_web3, y_web3

    # %%
    flipside_api_key = os.getenv('FLIPSIDE_API_KEY')
    alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
    opensea_api_key = os.getenv('OPENSEA_API_KEY')

    print(alchemy_api_key)

    data_dict = pull_data(api=api)

    print(f'pulled data')


    domain_rankings = pd.read_csv('../data/tranco_5863N.csv')
    google_rank = pd.DataFrame({'rank': [1], 'domain': ['google.com']})
    domain_rankings.columns = ['rank','domain']

    # Concatenate the new row with the original domain rankings
    domain_rankings = pd.concat([google_rank, domain_rankings], ignore_index=True)

    # Reset the index and display the updated rankings
    domain_rankings.reset_index(drop=True, inplace=True)

    if web2_data == True:

        print(f'web2 data')


        domain_path = '../data/domain-name-sales.tsv'  
        domain_data = pd.read_csv(domain_path, delimiter='\t')

        # %%
        domain_data.set_index('date', inplace=True)
        domain_data = domain_data.drop(columns=['venue'])
        domain_data.sort_index(inplace=True)
        domain_data

        domain_data = domain_data.reset_index()
        domain_data = domain_data.rename(columns={"date":"dt","price":"price_usd"})
        domain_data['dt'] = pd.to_datetime(domain_data['dt'])
        domain_data['dt'] = domain_data['dt'].dt.tz_localize('UTC')
        domain_data['dt'] = pd.to_datetime(domain_data['dt'])

        domain_data['web3'] = False

        namebio_path = '../data/namebio_sales.csv'
        namebio_data = pd.read_csv(namebio_path)
        namebio_data.set_index('Date', inplace=True)
        namebio_data = namebio_data.drop(columns=['Venue'])
        namebio_data.sort_index(inplace=True)

        namebio_data = namebio_data.reset_index()
        namebio_data = namebio_data.rename(columns={"Date":"dt","Price":"price_usd","Domain":"domain"})
        namebio_data['dt'] = pd.to_datetime(namebio_data['dt'])
        namebio_data['dt'] = namebio_data['dt'].dt.tz_localize('UTC')
        namebio_data['dt'] = pd.to_datetime(namebio_data['dt'])

        namebio_data['web3'] = False

        domain_data = pd.concat([domain_data,namebio_data],ignore_index=True)

    optimism_name_service_data= data_dict['optimism_name_service_data']
    optimism_name_service_data = optimism_name_service_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
    optimism_name_service_data

    Three_DNS_data = data_dict['Three_DNS_data']
    Three_DNS_data = Three_DNS_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
    Three_DNS_data
  
    optimistic_domains = data_dict['optimistic_domains']

    ens_data = data_dict['ens']
    ens_data = ens_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]

    
    unstoppable_sales_data = data_dict['unstoppable_sales_data']
    unstoppable_sales_data = unstoppable_sales_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
    unstoppable_sales_data

    
    base_domains_metadata_pd = data_dict['base_domains_metadata_pd']
    base_domains_metadata_pd = base_domains_metadata_pd[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
    base_domains_metadata_pd

    prices_data = data_dict['prices_data']
    prices_data = prices_data.dropna()
    prices_data['symbol'] = prices_data['symbol'].replace('WETH', 'ETH')


    prices_data = prices_data.pivot(index='dt',columns='symbol',values='price')
    prices_data = prices_data.reset_index()
    prices_data

    optimistic_domains_sales = data_dict['optimistic_domains_sales']

    print(f'optimistic_domains_sales: {optimistic_domains_sales}')

    # %%
    optimistic_domains_sales = optimistic_domains_sales.dropna(subset=['tokenid'])
    optimistic_domains_sales['tokenid']

    # %%
    optimistic_domains_sales['tokenid'] = optimistic_domains_sales['tokenid'].astype(int)
    optimistic_domains_sales.rename(columns={"tokenid":"tokenId"}, inplace=True)

    # %%
    optimistic_domains_sales['tokenId']

    # %%
    optimistic_domains['tokenId']


    # %%
    optimistic_data = pd.merge(optimistic_domains_sales, optimistic_domains, on='tokenId', how='left')
    optimistic_data.rename(columns={"tokenId":"nft_identifier","name":"nft_name", "day":"dt"}, inplace=True)

    # %%
    prices_data

    # %%
    optimism_name_service_data['dt'] = pd.to_datetime(optimism_name_service_data['dt'], unit='ms')
    Three_DNS_data['dt'] = pd.to_datetime(Three_DNS_data['dt'], unit='ms')
    ens_data['dt'] = pd.to_datetime(ens_data['dt'])
    unstoppable_sales_data['dt'] = pd.to_datetime(unstoppable_sales_data['dt'])
    base_domains_metadata_pd['dt'] = pd.to_datetime(base_domains_metadata_pd['dt'], unit='ms')

    def hourly(df):
        df['dt'] = df['dt'].dt.strftime('%Y-%m-%d %H-00-00')
        df['dt'] = pd.to_datetime(df['dt'])
        return df


    # %%
    Three_DNS_data = hourly(Three_DNS_data)
    optimism_name_service_data = hourly(optimism_name_service_data)
    ens_data = hourly(ens_data)
    unstoppable_sales_data = hourly(unstoppable_sales_data)
    base_domains_metadata_pd = hourly(base_domains_metadata_pd)

    print(f'standardized time')


    Three_DNS_data

    # %%
    Three_DNS_data['dt']

    # %%
    prices_data['dt'] = pd.to_datetime(prices_data['dt'])
    # prices_data.rename(columns={'DT':'dt'}, inplace=True)


    # %%
    print(f'prices data dt: {prices_data["dt"]}')
    # prices_data['dt'] = prices_data['dt'].dt.tz_convert('UTC')
    prices_data

    # %%
    Three_DNS_data = Three_DNS_data.merge(prices_data, how='left', on='dt')
    Three_DNS_data['price_usd'] = Three_DNS_data['token_amt_clean'] * Three_DNS_data['ETH']
    Three_DNS_data

    # %%
    optimism_name_service_data = optimism_name_service_data.merge(prices_data, how='left', on='dt')
    optimism_name_service_data['price_usd'] = optimism_name_service_data['token_amt_clean'] * optimism_name_service_data['ETH']
    optimism_name_service_data


    # %%
    ens_data = ens_data.merge(prices_data, how='left', on='dt')
    ens_data['price_usd'] = ens_data['token_amt_clean'] * ens_data['ETH']

    unstoppable_sales_data = unstoppable_sales_data.merge(prices_data, how='left', on='dt')
    unstoppable_sales_data['price_usd'] = unstoppable_sales_data['token_amt_clean'] * unstoppable_sales_data['ETH']

    # %%
    base_domains_metadata_pd = base_domains_metadata_pd.merge(prices_data, how='left', on='dt')
    base_domains_metadata_pd['price_usd'] = base_domains_metadata_pd['token_amt_clean'] * base_domains_metadata_pd['ETH']
    base_domains_metadata_pd

    print(f'optimistic_data:{optimistic_data}')

    # %%
    # optimistic_data = optimistic_data.merge(prices_data, how='left', on='dt')
    # optimistic_data['price_usd'] = optimistic_data['price'] * ens_data['ETH']
    optimistic_data.rename(columns={'price':'token_amt_clean'}, inplace=True)

    # %%
    optimistic_data['dt'] = pd.to_datetime(optimistic_data['dt'])
    # optimistic_data['dt'] = optimistic_data['dt'].dt.tz_localize('UTC')
    # optimistic_data['dt'] = pd.to_datetime(optimistic_data['dt'])

    print(f'optimistic_data:{optimistic_data}')

    

    # %%
    combined_dataset = pd.concat([
        ens_data[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        optimistic_data[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        optimism_name_service_data[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        unstoppable_sales_data[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        base_domains_metadata_pd[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        Three_DNS_data[['dt','nft_name','price_usd','token_amt_clean']].dropna()
    ], ignore_index=True)

    print(f'combined dataset')

    combined_dataset = combined_dataset.drop_duplicates()
    combined_dataset['dt'] = pd.to_datetime(combined_dataset['dt'], errors='coerce')
    combined_dataset = combined_dataset.sort_values(by='dt')
    combined_dataset = combined_dataset.reset_index(drop=True)
    combined_dataset


    # %%

    target = 'price_usd'
    combined_dataset = combined_dataset.rename(columns={'nft_name':'domain'})
    combined_dataset = combined_dataset.drop(columns=['token_amt_clean'])
    combined_dataset['web3'] = True

    # web3_data = combined_dataset.copy()

    if web2_data == True:
        print(f'adding web2 data')
        combined_dataset = pd.concat([combined_dataset, domain_data], ignore_index=True)

    print(f'domain feature engineering')

    combined_dataset['domain_length'] = combined_dataset['domain'].apply(len)
    print(f'@ num_vols')
    combined_dataset['num_vowels'] = combined_dataset['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
    combined_dataset['num_consonants'] = combined_dataset['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
    print(f'extracting tld')
    combined_dataset['tld'] = combined_dataset['domain'].apply(lambda x: x.split('.')[-1])  # Extract TLD
    combined_dataset['word_count'] = combined_dataset['domain'].apply(lambda x: len(x.split('-')))
    combined_dataset['has_numbers'] = combined_dataset['domain'].apply(lambda x: any(char.isdigit() for char in x))
    combined_dataset['tld_length'] = combined_dataset['tld'].apply(len)
    print(f'is_brandable')
    combined_dataset['is_brandable'] = combined_dataset['domain'].apply(is_brandable)
    combined_dataset['levenshtein_distance'] = combined_dataset['domain'].apply(min_levenshtein_distance)
    combined_dataset['is_subdomain'] = combined_dataset['domain'].apply(is_subdomain)
    print(f'domain_entropy')
    combined_dataset['domain_entropy'] = combined_dataset['domain'].apply(entropy)
    combined_dataset = add_domain_rank(combined_dataset, domain_rankings)

    # print(f'web3 domain feature engineering')

    # web3_data['domain_length'] = web3_data['domain'].apply(len)
    # web3_data['num_vowels'] = web3_data['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
    # web3_data['num_consonants'] = web3_data['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
    # web3_data['tld'] = web3_data['domain'].apply(lambda x: x.split('.')[-1])  # Extract TLD
    # web3_data['word_count'] = web3_data['domain'].apply(lambda x: len(x.split('-')))
    # web3_data['has_numbers'] = web3_data['domain'].apply(lambda x: any(char.isdigit() for char in x))
    # web3_data['tld_length'] = web3_data['tld'].apply(len)
    # web3_data['is_brandable'] = web3_data['domain'].apply(is_brandable)
    # web3_data['levenshtein_distance'] = web3_data['domain'].apply(min_levenshtein_distance)
    # web3_data['is_subdomain'] = web3_data['domain'].apply(is_subdomain)
    # web3_data['domain_entropy'] = web3_data['domain'].apply(entropy)
    # web3_data = add_domain_rank(web3_data, domain_rankings)

    if tld_weight != None:
        tld_weights = {
            'com': tld_weight['com'],
            'net': tld_weight['net'],
            'org': tld_weight['org'],
            'box': tld_weight['box'],
            'eth': tld_weight['eth']
            # Add more TLDs and their weights as needed
        }
        
        default_tld_weight = 1
        
        combined_dataset['tld_weight'] = combined_dataset['tld'].map(tld_weights).fillna(default_tld_weight)  # Default weight is 1 if tld is not in tld_weights
        # web3_data['tld_weight'] = combined_dataset['tld'].map(tld_weights).fillna(default_tld_weight) 

    if temporals == True:
        print(f'adding temporals')
        def calculate_rolling_statistics(df, col='price_usd'):
            # Rolling averages
            df['7d_rolling_avg_price'] = df[col].rolling(window=7).mean().fillna(0)
            # df['30d_rolling_avg_price'] = df[col].rolling(window=30).mean().fillna(0)

            # Rolling sum (sales volume)
            df['7d_sales_volume'] = df[col].rolling(window=7).sum().fillna(0)
            # df['30d_sales_volume'] = df[col].rolling(window=30).sum().fillna(0)

            # Expanding cumulative mean
            # df['cumulative_rolling_avg_price'] = df[col].expanding().mean()

            # Rolling count (domains sold)
            # df['7d_domains_sold'] = df[col].rolling(window=7).count().fillna(0)
            # df['30d_domains_sold'] = df[col].rolling(window=30).count().fillna(0)
            # df['60d_domains_sold'] = df[col].rolling(window=60).count().fillna(0)
            # df['90d_domains_sold'] = df[col].rolling(window=90).count().fillna(0)

            # Rolling standard deviation
            df['7d_rolling_std_dev'] = df[col].rolling(window=7).std().fillna(0)
            # df['30d_rolling_std_dev'] = df[col].rolling(window=30).std().fillna(0)

            # Rolling median
            # df['7d_rolling_median_price'] = df[col].rolling(window=7).median().fillna(0)
            # df['30d_rolling_median_price'] = df[col].rolling(window=30).median().fillna(0)

            # # Expanding cumulative sum (sales volume)
            # df['cumulative_sum_sales_volume'] = df[col].expanding().sum().fillna(0)
            
            return df
        
        combined_dataset = calculate_rolling_statistics(combined_dataset,'price_usd')
        # web3_data = calculate_rolling_statistics(web3_data,'price_usd')
    
    if correlation_analysis:
        print(f'correlation analysis')

        numeric_df = combined_dataset.select_dtypes(include=[float, int])
        # web3_data_numeric = web3_data.select_dtypes(include=[float, int])

        correlation_matrix = numeric_df.corr(corr_type)
        # web3_correlation_matrix = web3_data_numeric.corr(corr_type)

        target_corr = correlation_matrix[target].abs()
        # target_corr_web3 = web3_correlation_matrix[target].abs()
        print(f'threshold: {threshold}')

        print(f'target corr: {target_corr}')

        print(f'target corr under: {target_corr[target_corr < threshold]}')

    if threshold != None:
        columns_to_drop = target_corr[target_corr < threshold].index
        prophet_columns_to_drop = columns_to_drop.copy() + 'dt'
    elif threshold == None:
        columns_to_drop = ['90d_domains_sold', '60d_domains_sold', '30d_domains_sold', '7d_domains_sold', 'cumulative_sum_sales_volume',
                    'cumulative_rolling_avg_price','30d_rolling_median_price','7d_rolling_median_price',
                    '30d_rolling_std_dev']
        
        # Latest correlation values: most significant correlations among temporal vals
        # 7d_sales_volume                 0.378099
        # 7d_rolling_avg_price            0.378099
        # 7d_rolling_std_dev              0.378011

        prophet_columns_to_drop = ['dt','90d_domains_sold', '60d_domains_sold', '30d_domains_sold', '7d_domains_sold', 'cumulative_sum_sales_volume',
                        'cumulative_rolling_avg_price','30d_rolling_median_price','7d_rolling_median_price',
                        '30d_rolling_std_dev']

    combined_dataset = combined_dataset.drop_duplicates()
    combined_dataset['dt'] = pd.to_datetime(combined_dataset['dt'], errors='coerce')
    combined_dataset = combined_dataset.sort_values(by='dt')
    combined_dataset = combined_dataset.reset_index(drop=True)

    combined_dataset.to_csv('../data/last_dataset.csv')

    features = combined_dataset.drop(columns=target).columns
    print(f'target:{target},\n features:{features}')

    if columns_to_drop is None:
        columns_to_drop = []

    columns_to_drop = pd.Index(columns_to_drop)  # Ensure it's array-like
    prophet_columns_to_drop = pd.Index(prophet_columns_to_drop)
    # Drop columns from Index
    gen_features = features.difference(columns_to_drop)
    print(f'gen features {gen_features}')
    gen_features

    # %%
    prophet_features = features.difference(prophet_columns_to_drop)
    print(f'prophet features {prophet_features}')
    prophet_features

    # %%
    X = combined_dataset[gen_features]
    y = combined_dataset[target]

    print(f'X: {X} \ny: {y}')

    web3_data = combined_dataset[combined_dataset['web3']==True]

    X_web3 = web3_data[gen_features]
    y_web3 = web3_data[target]

    print(f'done process_data')

    return X, y, prophet_features, gen_features, target, combined_dataset, features, web3_data, X_web3, y_web3