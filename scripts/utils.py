import pandas as pd
import numpy as np
import random
from flipside import Flipside
import Levenshtein as lev
import math

domain_rankings = pd.read_csv('data/tranco_5863N.csv')
google_rank = pd.DataFrame({'rank': [1], 'domain': ['google.com']})
domain_rankings.columns = ['rank','domain']

# Concatenate the new row with the original domain rankings
domain_rankings = pd.concat([google_rank, domain_rankings], ignore_index=True)

# Reset the index and display the updated rankings
domain_rankings.reset_index(drop=True, inplace=True)

domain_rankings

top_rankings = domain_rankings['domain'].head(10000).values

def flipside_api_results(query, api_key):
  
  flipside_api_key = api_key
  flipside = Flipside(flipside_api_key, "https://api-v2.flipsidecrypto.xyz")

  query_result_set = flipside.query(query)
  # what page are we starting on?
  current_page_number = 1

  # How many records do we want to return in the page?
  page_size = 1000

  # set total pages to 1 higher than the `current_page_number` until
  # we receive the total pages from `get_query_results` given the 
  # provided `page_size` (total_pages is dynamically determined by the API 
  # based on the `page_size` you provide)

  total_pages = 2


  # we'll store all the page results in `all_rows`
  all_rows = []

  while current_page_number <= total_pages:
    results = flipside.get_query_results(
      query_result_set.query_id,
      page_number=current_page_number,
      page_size=page_size
    )

    total_pages = results.page.totalPages
    if results.records:
        all_rows = all_rows + results.records
    
    current_page_number += 1

  return pd.DataFrame(all_rows)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)

# Define a list of brandable words
brandable_words = ['google', 'apple', 'amazon', 'zoom', 'meta', 'coin', 'chain','uniswap',
                   'aave','tesla','ethereum','bitcoin','solana','token','news','apple',
                   'amazon','ai','tech','shop','cloud','data','world','chainlink']  # Add more as needed

# Function to check if the domain contains any brandable word
def is_brandable(domain):
    domain_name = domain.lower().split('.')[0]  # Get the domain part without TLD
    if domain_name == 'google':
        print(f'domain name: {domain_name}')
    for word in brandable_words:
        if domain_name == 'google' and word =='google':
            print(f'word: {word}')
        if word in domain_name:
            return 1  # Brandable
    return 0  # Not brandable

popular_domains = top_rankings

def min_levenshtein_distance(domain):
    domain_name = domain.lower().split('.')[0]  # Get the domain part without TLD
    distances = [lev.distance(domain_name, popular_domain.split('.')[0]) for popular_domain in popular_domains]
    return min(distances)  # Return the minimum distance

def is_subdomain(domain):
    # Split the domain by '.'
    domain_parts = domain.split('.')
    
    # If there are more than two parts (like 'sub.domain.com'), it's a subdomain
    if len(domain_parts) > 2:
        return 1  # It's a subdomain
    else:
        return 0  # It's not a subdomain
    
def entropy(domain):
    probabilities = [float(domain.count(c)) / len(domain) for c in set(domain)]
    return -sum([p * math.log(p, 2) for p in probabilities])

tld_categories = {
    'com': 'general',
    'tech': 'technology',
    'shop': 'e-commerce',
    'edu': 'education',
    'org': 'non-profit',
    'net': 'networking',
    'io': 'technology',
    'ai': 'artificial intelligence',
    'cloud': 'cloud computing',
    'data': 'data science',
    'news': 'news and media',
    'eth': 'crypto',
    'btc': 'crypto',
    'sol': 'crypto',
    'us': 'country code',
    'uk': 'country code',
    'ca': 'country code'
    # Add more TLD categories as needed
}

def get_domain_category(tld):
    return tld_categories.get(tld, 'unknown')

def add_domain_rank(combined_dataset, domain_rankings):
    # Create a dictionary from the rankings for faster lookup
    domain_rank_dict = dict(zip(domain_rankings['domain'], domain_rankings['rank']))
    
    # Add a new column to the dataset with the rank or 'unranked'
    combined_dataset['rank'] = combined_dataset['domain'].apply(
        lambda x: domain_rank_dict.get(x, 'unranked')
    )
    
    return combined_dataset


# Apply the function to your dataset
# combined_dataset['domain_category'] = combined_dataset['tld'].apply(get_domain_category)

# combined_dataset['levenshtein_distance'] = combined_dataset['domain'].apply(min_levenshtein_distance)

# Apply the function to the dataset
# combined_dataset['is_brandable'] = combined_dataset['domain'].apply(is_brandable)
