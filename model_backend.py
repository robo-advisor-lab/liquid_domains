import pandas as pd
import numpy as np
import random
import os
import sys
import requests
import time
import datetime as dt
from diskcache import Cache
import joblib

from plotly.utils import PlotlyJSONEncoder

from flask import Flask, request, jsonify, render_template

from dotenv import load_dotenv
from flipside import Flipside
from prophet import Prophet
import json

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

from scripts.utils import flipside_api_results, set_random_seed
from scripts.data_processing import process_data
from models.forecasters import EnsemblePredictor, Prophet_Domain_Valuator, Domain_Valuator, train_ridge_model, train_randomforest_model, train_prophet_model

from scripts.vizualizations import create_visualizations
from scripts.blockscout_visualizations import blockscout

from pyngrok import ngrok, conf, installer
import ssl

import sqlite3

from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3, EthereumTesterProvider
from web3.middleware import construct_sign_and_send_raw_middleware

pd.options.display.float_format = '{:,.2f}'.format

seed = 20

# Create a default SSL context that bypasses certificate verification
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

# Set the path to the ngrok executable installed by Chocolatey
ngrok_path = "C:\\ProgramData\\chocolatey\\bin\\ngrok.exe"

# Update the pyngrok configuration with the ngrok path
pyngrok_config = conf.PyngrokConfig(ngrok_path=ngrok_path)

# Check if ngrok is installed at the specified path, if not, install it using the custom SSL context
if not os.path.exists(pyngrok_config.ngrok_path):
    installer.install_ngrok(pyngrok_config.ngrok_path, context=context)

# Configure ngrok with custom SSL context
conf.set_default(pyngrok_config)
conf.get_default().ssl_context = context

load_dotenv()

ngrok_token = os.getenv('ngrok_token')

# Set your ngrok auth token
ngrok.set_auth_token(ngrok_token)

# Start ngrok
# public_url = ngrok.connect(5555, pyngrok_config=pyngrok_config, hostname="www.optimizerfinance.com").public_url
# public_url = ngrok.connect(5555, pyngrok_config=pyngrok_config).public_url
# print("ngrok public URL:", public_url)

with open(r'E:\Projects\liquid_domains\abi\ens_abi.json') as f:
    contract_abi = json.load(f)

with open(r'E:\Projects\liquid_domains\abi\ens_resolver_abi.json') as f:
    ENS_RESOLVER_ABI = json.load(f)

with open(r'E:\Projects\liquid_domains\abi\ens_registry_abi.json') as f:
    ENS_REGISTRY_ABI = json.load(f)

def network(chain='optimism-sepolia'):
    if chain == 'base-sepolia':
        w3 = Web3(Web3.HTTPProvider(f'https://{chain}.g.alchemy.com/v2/6AUlaGmWe505S7gRPZXVh4YEFgJdYHy5'))
    elif chain == 'Ethereum Mainnet':
        w3 = Web3(Web3.HTTPProvider(f'https://eth-mainnet.g.alchemy.com/v2/6AUlaGmWe505S7gRPZXVh4YEFgJdYHy5'))
    elif chain =='celo-dango':
        w3 = Web3(Web3.HTTPProvider(f'https://forno.dango.celo-testnet.org/'))
    elif chain =='mode-sepolia':
        w3 = Web3(Web3.HTTPProvider(f'https://sepolia.mode.network/'))
    else:
        w3 = Web3(Web3.HTTPProvider(f'https://optimism-sepolia.infura.io/v3/22b286f565734e3e80221a4212adc370'))

    print(f'Web3 provider URL: {w3.provider.endpoint_uri}')

    try:
        latest_block = w3.eth.get_block('latest')
        print(f'Latest block number: {latest_block["number"]}')
    except Exception as e:
        print(f'Error fetching latest block: {str(e)}')

    if chain == 'base-sepolia':
        contract_address = '0x571d1bd9F88Cd5cf8ae3d72Ca5fA06D593490869'
    elif chain == 'mode-sepolia':
        contract_address = '0x3E6f168587f9721A31f2FA1a560e6ab36d3B8c69'
    elif chain == 'celo-dango': 
        contract_address = '0x3E6f168587f9721A31f2FA1a560e6ab36d3B8c69'
    else:
        contract_address = '0x26dd9e35C36249907D6F63C1424BC2a44898600b'
    return w3, contract_address

# def search(domain, tld, liquidity_discount=0.3,ensamble_method='median'):
#     web3_tlds = ['.op', '.crypto', '.box', '.eth', '.base', '.nft', '.wallet', '.coin', '.finance', '.xyz', '.dao', '.bitcoin', '.blockchain']

#     print(f'domain: {domain}')
#     print(f'tld: {tld}')

#     # Allow subdomains, so no error for dot in domain
#     # Check if the TLD ends with any of the Web3 TLDs
#     if not any(tld.endswith(web3_tld) for web3_tld in web3_tlds):
#         raise ValueError('Error: Cannot Value Non-Web3 Domain')

#     full_domain = domain + tld
#     value = main(domain=full_domain, prophet_model=prophet_model, ridge_model=ridge_model, randomforest_model=randomforest_model, combined_dataset=combined_dataset, prophet_features=prophet_features, gen_features=gen_features, features=features, X=X, y=y,ensamble_method=ensamble_method)

#     discounted_value = value * liquidity_discount
#     liquidity_discount_filtered = 1 - liquidity_discount

#     print(f'Domain: {full_domain} \nValue: ${value:,.2f} \nLiquidity Discount: {liquidity_discount_filtered*100:.2f}% \nDiscounted Price w/ Liquidity Discount: ${discounted_value:,.2f}')
    
#     return value, discounted_value

w3, contract_address = network()
    
# # Initialize Web3
# w3 = Web3(Web3.HTTPProvider('https://optimism-sepolia.infura.io/v3/22b286f565734e3e80221a4212adc370'))
# print(f'Web3 provider URL: {w3.provider.endpoint_uri}')

# # Check if the connection is working
# try:
#     latest_block = w3.eth.get_block('latest')
#     print(f'Latest block number: {latest_block["number"]}')
# except Exception as e:
#     print(f'Error fetching latest block: {str(e)}')

# # print(f'Web3 version: {w3.__version__}')

# Load contract ABI and address
# with open('liquid-domains-abi.json') as f:
#     contract_abi = json.load(f)
#     print(f'Contract ABI loaded: {contract_abi}')

# # contract_address = '0x26dd9e35C36249907D6F63C1424BC2a44898600b'
# print(f'Contract address: {contract_address}')

# # Initialize the contract
# try:
#     contract = w3.eth.contract(address=contract_address, abi=contract_abi)
#     print('Contract initialized successfully.')
# except Exception as e:
#     print(f'Error initializing contract: {str(e)}')

# # Get the private key and create an account
private_key = os.getenv('DEPLOYER_PRIVATE_KEY')
account = Account.from_key(private_key)

# # Extract address and convert to checksum address
account_address = account.address
account_checksum = Web3.to_checksum_address(account_address)

# # Set the default account for Web3
w3.eth.default_account = account_checksum

# # Add middleware for signing and sending transactions
w3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))

print(f"Your hot wallet address is {account_address}")
print(f'Connected to {w3.eth.default_account} on optimism-sepolia')

ENS_CONTRACT_ADDRESS = os.getenv('ENS_CONTRACT_ADDRESS')

ENS_REGISTRY_ADDRESS = os.getenv('ENS_REGISTRY_ADDRESS')

API_KEY = os.getenv('FRONTEND_API_KEY')

print(f'api key: {API_KEY}')

api = False #Pulls the most recent data from all web3 sources, web2 sources are still via csv (This function takes 2+ hours, use it periodically)
web2_data = True #Includes web2 data in training
threshold = None #Correlation value for correlation analysis
temporals = True #Tends to inflate estimation
fine_tuning_web3 = True #Train again on web3 data only
correlation_analysis = False #Performs Correlation Analysis to 'price' col and drops cols under the threshold parameter 
last_dataset = True #Saves time of feature engineering, saves the latest copy of the dataset generated by the function

def check_api_key(request):
    api_key = request.headers.get('api_key')
    if api_key is None or api_key != f'Bearer {API_KEY}':
        return False
    return True

# Create Flask app
def create_app():
    app = Flask(__name__)

    cache = Cache('cache_dir')

    global historical_data
    historical_data = pd.DataFrame()
    historical_data = cache.get('historical_data', pd.DataFrame())

    # Load models
    global prophet_model, ridge_model, randomforest_model, cumulative_sales_chart, ma_plot, sold_domains_fig, rolling_avg_plot, combined_dataset

    X, y, prophet_features, gen_features, target, combined_dataset, features, _, _, _ = process_data(api=api,web2_data=web2_data,threshold=threshold,temporals=temporals,correlation_analysis=correlation_analysis,last_dataset=last_dataset)

    prophet_model = joblib.load('pkl/prophet_model.pkl')
    ridge_model = joblib.load('pkl/ridge_model.pkl')
    randomforest_model = joblib.load('pkl/randomforest_model.pkl')

    @app.route('/update_data')
    def update_data():
        X, y, prophet_features, gen_features, target, combined_dataset, features = process_data()
        return X, y, prophet_features, gen_features, target, combined_dataset, features
    # cumulative_sales_chart, ma_plot, sold_domains_fig, rolling_avg_plot = create_visualizations(combined_dataset)

    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/clear_cache', methods=['POST'])
    def clear_cache():
        cache.clear()
        return jsonify({"message": "Cache cleared successfully"})
    
    def get_metadata(domain):
        conn = sqlite3.connect('metadata.db')
        cursor = conn.cursor()
        cursor.execute('SELECT json_data FROM metadata WHERE domain = ?', (domain,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    
    @app.route('/metadata/<domain>', methods=['GET'])
    def metadata(domain):
        data = get_metadata(domain)
        if data:
            return jsonify({"domain": domain, "metadata": data}), 200
        else:
            return jsonify({"error": "Metadata not found"}), 404

    @app.route('/api/visualizations', methods=['GET'])
    def visualizations():
        global combined_dataset
        time_frame = request.args.get('time_frame', '7d')  # Default to 'all' if not provided
        print(f'timeframe {time_frame}')
        print(f'combined_data: {combined_dataset}')
        filtered_charts = create_visualizations(combined_dataset, time_frame)
        cached_data = {
            "cumulative_sales_chart": filtered_charts[0],
            "ma_plot": filtered_charts[1],
            "sold_domains_fig": filtered_charts[2],
            "rolling_avg_plot": filtered_charts[3]
        }
        return jsonify(cached_data)

    @app.route('/api/historical_data')
    def get_historical_data():
        global historical_data
        historical_data = pd.DataFrame()
        historical_data = cache.get('historical_data', pd.DataFrame())
        print(f'historical data at get historical_data: {historical_data}')
        historical_data_json = historical_data.to_dict(orient='records')
        return jsonify(historical_data_json)

    @app.route('/api/evaluate', methods=['POST'])
    def evaluate():
        data = request.get_json()
        domain = data.get('domain')
        
        # Check if the domain is provided
        if not domain:
            return jsonify({'error': 'No domain provided'}), 400
        
        if '.' in domain.split('.')[0]:  # Ensure the prefix before the TLD doesn't contain a dot
            return jsonify({'error': "Invalid input. Domain prefix should not contain a '.' character."}), 400
        
        # Process the domain value
        value = main(domain, prophet_model, ridge_model, randomforest_model, combined_dataset, prophet_features, gen_features, features, X, y)
        
        # Record the current time and update historical data
        today = dt.datetime.now()
        # value = f"${value:,.2f}"
        value_info = {
            "dt": today,
            "domain": domain,
            "value": value
        }
        update_historical_data(value_info)
        print(f'historical data after value update: {historical_data}')
        
        # Return the domain value
        return jsonify({'domain': domain, 'value': value})

    @app.route('/onchainval')
    def onchain_val():
        return render_template('onchain_valuation.html')
    
    @app.route("/api/request-domain", methods=["POST"])
    def proxy_request():
        data = request.json
        domain = data.get("domain")
        network = data.get("network")

        if not domain:
            return jsonify({"error": "Domain is required"}), 400
        
        if not network:
            return jsonify({"error": "Network is required"}), 400
        
        # Proxy the request to the Node.js server
        try:
            response = requests.post("http://localhost:3000/api/request-domain", json={"domain": domain, "network": network})
            response.raise_for_status()  # Raise an error for bad HTTP status codes
            return jsonify(response.json()), response.status_code
        except requests.RequestException as e:
            # Handle errors in the request to the Node.js server
            print(f"Request error: {e}")
            return jsonify({"error": "Failed to proxy request"}), 500
    
    @app.route('/api/latest_valuations', methods=['GET'])
    def latest_valuations():
        global historical_data
        historical_data = pd.DataFrame()
        historical_data = cache.get('historical_data', pd.DataFrame())
        latest_valuations = historical_data.head(10) 

        # for record in latest_valuations:
        #     record['Value'] = f"${record['Value']:.2f}" # Get the last 10 rows
        # latest_valuations['Value'] = latest_valuations['Value'].apply(lambda x: f"${x:,.2f}")
        latest_valuations_json = latest_valuations.to_dict(orient='records')
        return jsonify(latest_valuations_json)

    def update_historical_data(live_comp):
        global historical_data
        new_data = pd.DataFrame([live_comp])
        historical_data = pd.concat([historical_data, new_data]).reset_index(drop=True)
        historical_data.drop_duplicates(subset='domain', keep='last', inplace=True)
        historical_data.sort_values(by='dt', ascending=False, inplace=True)
        cache.set('historical_data', historical_data)

    @app.route('/mint')
    def mint():
        return render_template('mint.html')
    
    @app.route('/endpoints')
    def endpoints():
        return render_template('onchainEndpoints.html')
    
    @app.route('/endpoint_visualizations')
    def endpoints_charts():
        filtered_charts = blockscout()
        cached_data = {
            "op-sepolia_fig": filtered_charts[0],
            "base-sepolia_fig": filtered_charts[1],
        }
        return jsonify(cached_data)
        
    @app.route('/api/mint', methods=['POST'])
    def api_mint():
        data = request.get_json()
        user_account = data.get('account')
        uri = data.get('uri')  # URI for metadata
        network_name = data.get('networkName')  # Network parameter

        # Log the received data
        print(f'Received data: account={account}, uri={uri}, network={network_name}')

        # Validate input data
        if not account or not uri or not network_name:
            return jsonify({"error": "Account address, URI, and network are required"}), 400
        
        # Convert account address to checksum format
        user_account_checksum = Web3.to_checksum_address(user_account)

        try:
            # Get Web3 provider and contract address for the selected network
            w3, contract_address = network(network_name)

            # Load the contract ABI (assuming this is already loaded)
            with open('liquid-domains-abi.json') as f:
                contract_abi = json.load(f)
            
            # Initialize the contract with the correct address and ABI
            contract = w3.eth.contract(address=contract_address, abi=contract_abi)

            w3.eth.default_account = account_checksum
            
            # Get the latest nonce for the account
            nonce = w3.eth.get_transaction_count(w3.eth.default_account)

            # Build the transaction
            tx = contract.functions.safeMint(user_account_checksum, uri).build_transaction({
                'from': w3.eth.default_account,
                'nonce': nonce,
                'gas': 2000000,
                'gasPrice': w3.eth.gas_price
            })

            # Sign the transaction
            signed_tx = w3.eth.account.sign_transaction(tx, private_key)

            # Send the transaction
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for transaction receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

            # Log the entire receipt for debugging
            print(f'Receipt: {receipt}')

            return jsonify({"message": "Minted successfully! Refresh page to see value.", "transactionHash": tx_hash.hex()})
        except Exception as e:
            # Log the exception
            print(f'Error: {str(e)}')
            return jsonify({"error": str(e)}), 400  
    
    @app.route('/api/domain_values', methods=['POST'])
    def api_domain_values():
        """Fetch ENS domains owned by a user and their valuations."""
        data = request.get_json()
        user_account = data.get('account')
        network_name = data.get('networkName')

        print(f'data: {data}')

        if not user_account:
            return jsonify({'error': 'Account address is required'}), 400

        user_address = Web3.to_checksum_address(user_account)

        # Connect to the Ethereum network
        w3, contract_address = network(network_name)

        # Load ABIs if they are JSON strings
        if isinstance(ENS_REGISTRY_ABI, str):
            ENS_REGISTRY_ABI = json.loads(ENS_REGISTRY_ABI)
        if isinstance(ENS_RESOLVER_ABI, str):
            ENS_RESOLVER_ABI = json.loads(ENS_RESOLVER_ABI)
        if isinstance(ENS_TOKEN_ABI, str):
            ENS_TOKEN_ABI = json.loads(ENS_TOKEN_ABI)

        # Initialize ENS Registry and Token contracts
        ens_registry = w3.eth.contract(address=ENS_REGISTRY_ADDRESS, abi=ENS_REGISTRY_ABI)
        ens_token = w3.eth.contract(address=ENS_TOKEN_ADDRESS, abi=ENS_TOKEN_ABI)

        try:
            # Get the number of ENS domains owned
            balance = ens_token.functions.balanceOf(user_address).call()
            domain_values = []
            total_value = 0

            print(f"User owns {balance} ENS domain(s)")

            for index in range(balance):
                # Get ENS token ID
                token_id = ens_token.functions.tokenOfOwnerByIndex(user_address, index).call()
                print(f'Token ID: {token_id}')

                # Get the ENS node (hash of the domain)
                node = Web3.solidity_keccak(['uint256'], [token_id])

                # Fetch resolver address for this ENS name
                resolver_address = ens_registry.functions.resolver(node).call()
                if resolver_address == "0x0000000000000000000000000000000000000000":
                    print(f"No resolver found for token ID {token_id}, skipping...")
                    continue

                # Load resolver contract
                resolver_contract = w3.eth.contract(address=resolver_address, abi=ENS_RESOLVER_ABI)

                # Fetch human-readable ENS name
                try:
                    domain_name = resolver_contract.functions.name(node).call()
                    print(f"Resolved domain: {domain_name}")

                    # Get valuation for domain name
                    value = get_domain_value(domain_name)
                    domain_values.append({'domain': domain_name, 'value': value})
                    total_value += value

                except Exception as e:
                    print(f"Error resolving domain for token {token_id}: {e}")
                    continue

            return jsonify({'domains': domain_values, 'totalValue': total_value})

        except Exception as e:
            print(f"Error fetching ENS domains: {e}")
            return jsonify({'error': str(e)}), 400

    def extract_domain_name_from_metadata(metadata_uri):
        # Extract everything after 'metadata' in the metadata URI
        # Example metadata_uri: 'metadatatest.eth'
        # We want to extract 'test.eth' from this

        # Find the position of 'metadata' in the URI
        prefix = 'metadata'
        start_index = metadata_uri.find(prefix)

        if start_index != -1:
            # Extract everything after 'metadata'
            domain_name_with_extension = metadata_uri[start_index + len(prefix):]
            return domain_name_with_extension
        
        return None

    def get_domain_value(domain_name):
        # Call the domain valuator API to get the value
        response = requests.post('http://localhost:5555/api/evaluate', json={'domain': domain_name, 'api_key': API_KEY})
        data = response.json()
        return data.get('value', 0)
    
    return app

# def main(domain, prophet_model, ridge_model, randomforest_model, combined_dataset, prophet_features, gen_features, features, X, y):
#     seed = 20
#     set_random_seed(seed)

#     # graph_json_1 = json.dumps(cumulative_sales_chart, cls=PlotlyJSONEncoder)
#     # graph_json_2 = json.dumps(ma_plot, cls=PlotlyJSONEncoder)
#     # graph_json_3 = json.dumps(sold_domains_fig, cls=PlotlyJSONEncoder)
#     # graph_json_4 = json.dumps(rolling_avg_plot, cls=PlotlyJSONEncoder)

#     prophet_features_data = combined_dataset.copy()
#     prophet_features_data.rename(columns={"dt": "ds", "price_usd": "y"}, inplace=True)

#     prophet_valuator = Prophet_Domain_Valuator(domain, prophet_features, prophet_features_data)
#     prophet_valuator.model_prep()
#     prophet_domain_value = prophet_valuator.value_domain(prophet_model)

#     features_data = combined_dataset.copy()
#     features_data['dt'] = features_data['dt'].dt.tz_localize(None)
#     features_data = features_data[features] 

#     ridge_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)
#     ridge_valuator.model_prep()
#     ridge_domain_value = ridge_valuator.value_domain(ridge_model)

#     randomforest_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)
#     randomforest_valuator.model_prep()
#     randomforest_domain_value = randomforest_valuator.value_domain(randomforest_model)

#     individual_predictions = [
#         prophet_domain_value,
#         ridge_domain_value,
#         randomforest_domain_value
#     ]

#     ensemble_domain_value = np.median(individual_predictions)
#     print(f'individual valuations: {individual_predictions}')
#     print(f'ensamble value: {ensemble_domain_value}')

#     return ensemble_domain_value

def main(domain, prophet_model, ridge_model, randomforest_model, combined_dataset, prophet_features, gen_features, features, X, y,seed=seed,ensamble_method = 'median'):
    set_random_seed(seed)

    prophet_features_data = combined_dataset.copy()
    prophet_features_data.rename(columns={"dt": "ds", "price_usd": "y"}, inplace=True)

    prophet_valuator = Prophet_Domain_Valuator(domain, prophet_features, prophet_features_data)
    prophet_valuator.model_prep()
    prophet_domain_value = prophet_valuator.value_domain(prophet_model)

    features_data = combined_dataset.copy()
    features_data['dt'] = features_data['dt'].dt.tz_localize(None)
    features_data = features_data[features] 

    ridge_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)
    ridge_valuator.model_prep()
    ridge_domain_value = ridge_valuator.value_domain(ridge_model)

    randomforest_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)
    randomforest_valuator.model_prep()
    randomforest_domain_value = randomforest_valuator.value_domain(randomforest_model)

    individual_predictions = [
        prophet_domain_value,
        ridge_domain_value,
        randomforest_domain_value
    ]

    if ensamble_method == 'mean':
        ensemble_domain_value = np.mean(individual_predictions)
    elif ensamble_method == 'median':
        ensemble_domain_value = np.median(individual_predictions)

    print(f'individual valuations: {individual_predictions}')
    print(f'ensamble value: {ensemble_domain_value}')

    if ensemble_domain_value < 0:
        print(f'Defaulting from Negative to $0')
        ensemble_domain_value = 0


    return ensemble_domain_value

if __name__ == '__main__':
    app = create_app()
    app.run(debug=False, port=5555)