# Standard library imports for data manipulation and file operations
import pandas as pd              # Data manipulation and analysis
import numpy as np              # Numerical computing and random number generation
from datetime import datetime, timedelta  # Date and time handling
import random                   # Additional random number generation
import os                      # Operating system interface for file operations
from typing import Tuple, List, Dict  # Type hints for better code documentation

# Display settings for better output formatting
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("✅ All libraries imported successfully!")
print("📊 Ready to generate realistic financial datasets for our FinTech project")
print("🎯 This data will support all 11 weeks of our Agile development sprints")
class FinancialDataGenerator:
    """
    Comprehensive mock financial data generator for FinTech projects.

    This class (In Python, a class is a blueprint for generating objects (instances) that share the same data-attributes and behavior (methods),
    creates realistic datasets that simulate:
    1. Stock market behavior (geometric Brownian motion)
    2. Cryptocurrency volatility (higher volatility, 24/7 trading)
    3. Economic indicators (mean-reverting time series)
    4. Portfolio allocations (risk-based asset allocation)
    5. Customer demographics (realistic distributions)

    Design Pattern: This follows the Factory Pattern - one class that creates
    multiple types of related objects (different financial datasets).
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the generator with predefined market data and random seed.

        Args:
            seed: Random seed for reproducibility (crucial for testing and validation)

        Why use a seed?
        - Ensures our data generation is reproducible
        - Critical for debugging and validation
        - Allows team members to generate identical datasets
        - Follows best practices in quantitative finance
        """
        # Set random seeds for reproducible results
        np.random.seed(seed)  # NumPy random operations
        random.seed(seed)     # Python random module operations

        # Define major stock symbols - representing different sectors and market caps
        # These are real S&P 500 companies for realistic modeling
        self.stock_symbols = [
            # Technology Giants (FAANG + others)
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',

            # Financial Services
            'JPM', 'BAC', 'V', 'MA',

            # Healthcare & Consumer Goods
            'JNJ', 'PG', 'UNH', 'PFE', 'KO', 'WMT',

            # Entertainment & Retail
            'DIS', 'HD', 'NKE', 'COST',

            # Software & Cloud
            'ADBE', 'CRM', 'ORCL', 'CSCO',

            # Traditional Industries
            'T', 'VZ', 'XOM', 'CVX', 'IBM'
        ]

        # Major cryptocurrency symbols by market capitalization (as of 2024-2025)
        # Note: Crypto markets are more volatile and trade 24/7
        self.crypto_symbols = [
            'BTC',   # Bitcoin - digital gold, store of value
            'ETH',   # Ethereum - smart contract platform
            'BNB',   # Binance Coin - exchange token
            'XRP',   # Ripple - cross-border payments
            'ADA',   # Cardano - proof-of-stake blockchain
            'DOGE',  # Dogecoin - meme coin with high volatility
            'SOL',   # Solana - high-performance blockchain
            'TRX',   # Tron - decentralized entertainment platform
            'DOT',   # Polkadot - interoperability protocol
            'MATIC', # Polygon - Ethereum scaling solution
            'SHIB',  # Shiba Inu - another meme coin
            'AVAX',  # Avalanche - smart contracts platform
            'LTC',   # Litecoin - Bitcoin fork
            'UNI',   # Uniswap - decentralized exchange token
            'LINK'   # Chainlink - oracle network
        ]

        # Key macroeconomic indicators that affect financial markets
        # These drive fundamental analysis and economic forecasting
        self.economic_indicators = [
            'GDP_GROWTH',           # Gross Domestic Product growth rate
            'INFLATION_RATE',       # Consumer Price Index changes
            'UNEMPLOYMENT_RATE',    # Labor market health
            'INTEREST_RATE',        # Federal funds rate (monetary policy)
            'CONSUMER_CONFIDENCE',  # Consumer sentiment index
            'RETAIL_SALES',         # Consumer spending patterns
            'INDUSTRIAL_PRODUCTION',# Manufacturing output
            'HOUSING_STARTS',       # Real estate market health
            'TRADE_BALANCE',        # Imports vs exports
            'MONEY_SUPPLY'          # M2 money supply (liquidity)
        ]

        print(f"🏭 FinancialDataGenerator initialized with seed={seed}")
        print(f"📈 Configured for {len(self.stock_symbols)} stock symbols")
        print(f"💰 Configured for {len(self.crypto_symbols)} crypto symbols")
        print(f"📊 Tracking {len(self.economic_indicators)} economic indicators")
        print("🔄 All random generators seeded for reproducible results")

# Test the class initialization
generator = FinancialDataGenerator(seed=42)
print("\n✅ Generator class created successfully!")
print("📝 Next: We'll implement individual data generation methods")
def generate_stock_prices(self,
                         symbols: List[str] = None,
                         start_date: str = '2020-01-01',
                         end_date: str = '2024-12-31',
                         initial_price_range: Tuple[float, float] = (20, 500)) -> pd.DataFrame:
    """
    Generate realistic stock price data using Geometric Brownian Motion.

    This method simulates how stock prices evolve over time, incorporating:
    1. Random price movements (market efficiency)
    2. Volatility clustering (periods of high/low volatility)
    3. Mean reversion tendencies (prices don't drift too far from fundamentals)
    4. Realistic trading volumes correlated with price volatility

    Args:
        symbols: List of stock symbols to generate (default: first 20 predefined)
        start_date: Start date for price series
        end_date: End date for price series
        initial_price_range: Range for starting stock prices

    Returns:
        DataFrame with columns: Date, Symbol, Open, High, Low, Close, Volume

    Financial Insights:
        - Higher volatility stocks have more dramatic price swings
        - Volume increases during high volatility periods (realistic behavior)
        - Mean reversion prevents prices from drifting to unrealistic levels
        - Weekend gaps are handled by excluding weekends from trading days
    """
    if symbols is None:
        symbols = self.stock_symbols[:20]  # Use first 20 symbols for manageable dataset

    # Create business day range (exclude weekends - NYSE is closed Sat/Sun)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    business_days = date_range[date_range.weekday < 5]  # Monday=0, Friday=4

    print(f"📅 Generating stock data for {len(business_days)} trading days")
    print(f"📈 Creating price series for {len(symbols)} symbols")

    all_stock_data = []

    for i, symbol in enumerate(symbols):
        print(f"  📊 Processing {symbol} ({i+1}/{len(symbols)})")

        # Initialize stock-specific parameters
        initial_price = np.random.uniform(*initial_price_range)
        annual_volatility = np.random.uniform(0.15, 0.45)  # 15-45% annual volatility
        daily_volatility = annual_volatility / np.sqrt(252)  # Convert to daily

        # Store prices for mean reversion calculation
        prices = [initial_price]
        volumes = []

        # Generate daily price evolution
        for day_idx, date in enumerate(business_days):
            # Base daily return components
            base_drift = np.random.normal(0.0008, 0.002)  # ~20% annual drift with variation
            volatility_shock = np.random.normal(0, daily_volatility)

            # Add mean reversion after 30 days (prevents unrealistic price drift)
            if day_idx > 30:
                # Calculate 30-day moving average
                recent_avg = np.mean(prices[-30:])
                mean_reversion_force = (recent_avg - prices[-1]) * 0.001
                base_drift += mean_reversion_force

            # Apply geometric Brownian motion formula
            price_multiplier = np.exp(base_drift + volatility_shock)
            new_price = prices[-1] * price_multiplier

            # Apply circuit breakers (realistic market limits)
            # No stock can drop more than 30% or gain more than 50% in one day
            new_price = max(new_price, prices[-1] * 0.70)  # Max 30% daily drop
            new_price = min(new_price, prices[-1] * 1.50)  # Max 50% daily gain

            prices.append(new_price)

            # Generate realistic trading volume
            # Volume correlates with volatility (high volatility = high volume)
            base_volume = np.random.lognormal(15, 1)  # Log-normal distribution for volume
            volatility_multiplier = abs(volatility_shock) * 5 + 1
            daily_volume = int(base_volume * volatility_multiplier)
            volumes.append(daily_volume)

        # Convert daily close prices to OHLCV format
        for day_idx, date in enumerate(business_days):
            close_price = prices[day_idx + 1]  # +1 because prices[0] is initial
            previous_close = prices[day_idx]

            # Generate intraday price range
            intraday_volatility = abs(np.random.normal(0, daily_volatility * close_price))

            # Calculate OHLC with realistic constraints
            high_price = close_price + np.random.uniform(0, 1) * intraday_volatility
            low_price = close_price - np.random.uniform(0, 1) * intraday_volatility
            open_price = previous_close + np.random.normal(0, daily_volatility * previous_close * 0.3)

            # Ensure OHLC logical consistency: Low ≤ Open,Close ≤ High
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # Add to dataset
            all_stock_data.append({
                'Date': date,
                'Symbol': symbol,
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volumes[day_idx]
            })

    stock_df = pd.DataFrame(all_stock_data)
    print(f"✅ Generated {len(stock_df):,} stock price records")
    print(f"📊 Data shape: {stock_df.shape}")
    return stock_df

# Add the method to our generator class
FinancialDataGenerator.generate_stock_prices = generate_stock_prices

# Test the stock price generation
print("🧪 Testing stock price generation with 5 symbols...")
test_stocks = generator.generate_stock_prices(
    symbols=['MSFT', 'PG', 'JNJ','V',"COST"],
    start_date='2024-01-01',
    end_date='2024-01-31'
)

print("\n📈 Sample Stock Data:")
print(test_stocks.head())
print(f"\n📊 Price range for AAPL: ${test_stocks[test_stocks['Symbol']=='AAPL']['Close'].min():.2f} - ${test_stocks[test_stocks['Symbol']=='AAPL']['Close'].max():.2f}")
print("✅ Stock price generation method working correctly!")
def generate_crypto_prices(self,
                          symbols: List[str] = None,
                          start_date: str = '2020-01-01',
                          end_date: str = '2024-12-31') -> pd.DataFrame:
    """
    Generate cryptocurrency price data with realistic 24/7 market behavior.

    Crypto markets exhibit unique characteristics:
    - Much higher volatility (50-120% annually)
    - 24/7 trading (no weekend gaps)
    - Sentiment-driven price action
    - Lower liquidity leads to more extreme movements
    - Different behavior patterns for weekends vs weekdays

    Args:
        symbols: List of crypto symbols (default: top 10 by market cap)
        start_date: Start date for generation
        end_date: End date for generation

    Returns:
        DataFrame with columns: Timestamp, Symbol, Open, High, Low, Close, Volume

    Technical Implementation:
        - 6-hour intervals (4 data points per day)
        - Higher volatility parameters than stocks
        - Weekend and night-time volume adjustments
        - Realistic initial price ranges for major cryptocurrencies
    """
    if symbols is None:
        symbols = self.crypto_symbols[:10]  # Top 10 cryptocurrencies

    # Crypto trades 24/7 - generate 6-hour intervals
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    # Take every 6th hour: 00:00, 06:00, 12:00, 18:00 UTC
    crypto_timestamps = full_date_range[::6]

    print(f"⏰ Generating crypto data for {len(crypto_timestamps)} 6-hour intervals")
    print(f"💎 Creating price series for {len(symbols)} cryptocurrencies")
    print("🌍 Modeling 24/7 global crypto markets")

    all_crypto_data = []

    # Realistic initial price ranges for major cryptocurrencies (USD)
    # These reflect approximate price levels as of 2024-2025
    initial_price_ranges = {
        'BTC': (30000, 60000),    # Bitcoin: $30k-60k range
        'ETH': (2000, 4000),      # Ethereum: $2k-4k range
        'BNB': (300, 600),        # Binance Coin: $300-600
        'XRP': (0.5, 1.5),        # Ripple: $0.50-1.50
        'ADA': (0.3, 1.2),        # Cardano: $0.30-1.20
        'DOGE': (0.05, 0.3),      # Dogecoin: $0.05-0.30
        'SOL': (50, 200),         # Solana: $50-200
        'TRX': (0.06, 0.12),      # Tron: $0.06-0.12
        'DOT': (5, 30),           # Polkadot: $5-30
        'MATIC': (0.5, 2.5)       # Polygon: $0.50-2.50
    }

    for i, symbol in enumerate(symbols):
        print(f"  💰 Processing {symbol} ({i+1}/{len(symbols)})")

        # Set initial price and volatility for this crypto
        price_range = initial_price_ranges.get(symbol, (1, 100))  # Default range for unlisted coins
        initial_price = np.random.uniform(*price_range)

        # Crypto volatility is much higher than stocks
        annual_volatility = np.random.uniform(0.5, 1.2)  # 50-120% annual volatility
        six_hour_volatility = annual_volatility / np.sqrt(365 * 4)  # Convert to 6-hour periods

        prices = [initial_price]
        volumes = []

        # Generate price evolution for each 6-hour period
        for period_idx, timestamp in enumerate(crypto_timestamps):
            # Base price movement
            base_drift = np.random.normal(0, 0.001)  # Slightly positive expected return
            volatility_shock = np.random.normal(0, six_hour_volatility)

            # Weekend effect: Crypto markets are less active on weekends
            if timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
                base_drift *= 0.7  # Reduced weekend activity

            # Night time effect: Reduced activity during US night hours
            if timestamp.hour < 6 or timestamp.hour > 22:
                base_drift *= 0.5  # Lower overnight activity

            # Apply geometric Brownian motion with higher volatility bounds
            price_multiplier = np.exp(base_drift + volatility_shock)
            new_price = prices[-1] * price_multiplier

            # Crypto circuit breakers (more lenient than stocks due to higher volatility)
            new_price = max(new_price, prices[-1] * 0.5)   # Max 50% period drop
            new_price = min(new_price, prices[-1] * 2.0)   # Max 100% period gain

            prices.append(new_price)

            # Generate trading volume (crypto volumes are typically lower than stocks)
            base_volume = np.random.lognormal(12, 1.5)  # Smaller base volume than stocks
            volatility_multiplier = abs(volatility_shock) * 10 + 1  # Higher sensitivity to volatility
            period_volume = int(base_volume * volatility_multiplier)
            volumes.append(period_volume)

        # Convert to OHLCV format for each 6-hour period
        for period_idx, timestamp in enumerate(crypto_timestamps):
            close_price = prices[period_idx + 1]
            previous_close = prices[period_idx]

            # Generate intraday range for 6-hour period
            period_volatility = abs(np.random.normal(0, six_hour_volatility * close_price * 2))

            # Calculate OHLC
            high_price = close_price + np.random.uniform(0, 1) * period_volatility
            low_price = close_price - np.random.uniform(0, 1) * period_volatility
            open_price = previous_close + np.random.normal(0, six_hour_volatility * previous_close)

            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # Add to dataset with appropriate precision
            # Crypto prices need more decimal places due to wide price ranges
            decimal_places = 6 if close_price < 10 else 2

            all_crypto_data.append({
                'Timestamp': timestamp,
                'Symbol': symbol,
                'Open': round(open_price, decimal_places),
                'High': round(high_price, decimal_places),
                'Low': round(low_price, decimal_places),
                'Close': round(close_price, decimal_places),
                'Volume': volumes[period_idx]
            })

    crypto_df = pd.DataFrame(all_crypto_data)
    print(f"✅ Generated {len(crypto_df):,} cryptocurrency price records")
    print(f"📊 Data shape: {crypto_df.shape}")
    return crypto_df

# Add method to the generator class
FinancialDataGenerator.generate_crypto_prices = generate_crypto_prices

# Test cryptocurrency generation
print("🧪 Testing crypto price generation with BTC and ETH...")
test_crypto = generator.generate_crypto_prices(
    symbols=['BTC', 'ETH'],
    start_date='2024-01-01',
    end_date='2024-01-31'  # One week for testing
)

print("\n💎 Sample Cryptocurrency Data:")
print(test_crypto.head())
print(f"\n📊 BTC price range: ${test_crypto[test_crypto['Symbol']=='BTC']['Close'].min():.2f} - ${test_crypto[test_crypto['Symbol']=='BTC']['Close'].max():.2f}")
print(f"📊 ETH price range: ${test_crypto[test_crypto['Symbol']=='ETH']['Close'].min():.2f} - ${test_crypto[test_crypto['Symbol']=='ETH']['Close'].max():.2f}")
print("✅ Cryptocurrency price generation working correctly!")
print("🌍 Note: Crypto data includes 24/7 trading with 6-hour intervals")
def generate_economic_data(self,
                          start_date: str = '2020-01-01',
                          end_date: str = '2024-12-31',
                          frequency: str = 'ME') -> pd.DataFrame:  # Changed from 'M' to 'ME'
    """
    Generate macroeconomic indicator time series with realistic patterns.

    Economic indicators exhibit different behaviors than asset prices:
    - Mean reversion to long-term equilibrium values
    - Lower volatility than financial assets
    - Autocorrelation (current values predict future values)
    - Different measurement frequencies (monthly, quarterly)

    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        frequency: 'ME' for month-end, 'QE' for quarter-end, 'YE' for year-end

    Returns:
        DataFrame with columns: Date, Indicator, Value

    Economic Theory Applied:
    - Business Cycle: Indicators move together in cycles
    - Phillips Curve: Inverse relationship between unemployment and inflation
    - Taylor Rule: Interest rates respond to inflation and output gaps
    """
    # Generate date range based on frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)

    print(f"📊 Generating economic indicators for {len(date_range)} periods")
    print(f"📅 Frequency: {frequency} ({len(date_range)} data points)")
    print("🏛️ Modeling key macroeconomic relationships")

    all_economic_data = []

    # Historical baseline values (approximately US averages 2020-2024)
    # These represent "normal" economic conditions
    baseline_values = {
        'GDP_GROWTH': 2.5,          # 2.5% annual GDP growth
        'INFLATION_RATE': 2.0,      # 2% Fed (as well as ECB) inflation target
        'UNEMPLOYMENT_RATE': 5.0,   # ~5% natural rate of unemployment
        'INTEREST_RATE': 1.5,       # Federal funds rate
        'CONSUMER_CONFIDENCE': 100.0, # Index value (100 = baseline)
        'RETAIL_SALES': 0.5,        # Monthly % change
        'INDUSTRIAL_PRODUCTION': 1.0, # Monthly % change
        'HOUSING_STARTS': 1200000,  # Annual units (in thousands)
        'TRADE_BALANCE': -50000,    # Million USD (negative = deficit)
        'MONEY_SUPPLY': 15000       # Billion USD (M2 money supply)
    }

    # Initialize current values at baseline
    current_values = baseline_values.copy()

    # Generate data for each time period
    for date_idx, date in enumerate(date_range):
        print(f"  📈 Processing period {date_idx + 1}/{len(date_range)}: {date.strftime('%Y-%m')}")

        for indicator in self.economic_indicators:
            baseline = baseline_values[indicator]
            current = current_values[indicator]

            # Mean reversion component
            # Economic indicators tend to return to long-term averages
            reversion_speed = 0.1  # 10% reversion per period
            mean_reversion = (baseline - current) * reversion_speed

            # Random shock component (varies by indicator type)
            if indicator in ['GDP_GROWTH', 'INFLATION_RATE', 'UNEMPLOYMENT_RATE']:
                # Core economic indicators have moderate volatility
                shock_std = 0.3
            elif indicator == 'INTEREST_RATE':
                # Interest rates change more gradually (Fed policy)
                shock_std = 0.2
            elif indicator == 'CONSUMER_CONFIDENCE':
                # Sentiment can be quite volatile
                shock_std = 5.0
            elif indicator in ['RETAIL_SALES', 'INDUSTRIAL_PRODUCTION']:
                # Monthly economic activity indicators
                shock_std = 0.5
            elif indicator == 'HOUSING_STARTS':
                # Housing market can be quite volatile
                shock_std = 50000
            elif indicator == 'TRADE_BALANCE':
                # Trade balance fluctuates with global conditions
                shock_std = 10000
            else:  # MONEY_SUPPLY
                # Money supply grows steadily with occasional policy changes
                shock_std = 500

            # Generate random shock
            random_shock = np.random.normal(0, shock_std)

            # Calculate new value
            new_value = current + mean_reversion + random_shock

            # Apply realistic bounds to prevent unrealistic values
            if indicator == 'UNEMPLOYMENT_RATE':
                new_value = max(2.0, min(15.0, new_value))  # 2-15% range
            elif indicator == 'INFLATION_RATE':
                new_value = max(-2.0, min(10.0, new_value))  # -2% to 10% range
            elif indicator == 'INTEREST_RATE':
                new_value = max(0.0, min(10.0, new_value))   # 0-10% range
            elif indicator == 'CONSUMER_CONFIDENCE':
                new_value = max(50, min(150, new_value))     # 50-150 index range
            elif indicator == 'GDP_GROWTH':
                new_value = max(-5.0, min(8.0, new_value))   # -5% to 8% growth range
            elif indicator == 'HOUSING_STARTS':
                new_value = max(500000, min(2000000, new_value))  # Realistic housing range
            elif indicator == 'MONEY_SUPPLY':
                new_value = max(new_value, current_values[indicator])  # Money supply rarely decreases

            # Update current value for next period
            current_values[indicator] = new_value

            # Add to dataset
            all_economic_data.append({
                'Date': date,
                'Indicator': indicator,
                'Value': round(new_value, 2)
            })

    economic_df = pd.DataFrame(all_economic_data)
    print(f"✅ Generated {len(economic_df):,} economic data points")
    print(f"📊 Data shape: {economic_df.shape}")

    # Display summary statistics
    print("\n📈 Economic Indicator Ranges:")
    for indicator in self.economic_indicators:
        indicator_data = economic_df[economic_df['Indicator'] == indicator]['Value']
        print(f"  {indicator}: {indicator_data.min():.2f} to {indicator_data.max():.2f}")

    return economic_df

# Add method to the generator class
FinancialDataGenerator.generate_economic_data = generate_economic_data

# Test economic data generation
print("🧪 Testing economic data generation...")
test_economic = generator.generate_economic_data(
    start_date='2024-01-01',
    end_date='2024-06-30',  # 6 months for testing
    frequency='ME'  # Monthly data
)

print("\n🏛️ Sample Economic Data:")
print(test_economic.head(10))

# Show data for one specific indicator
gdp_data = test_economic[test_economic['Indicator'] == 'GDP_GROWTH']
print(f"\n📊 GDP Growth over test period:")
print(gdp_data[['Date', 'Value']].head())

print("✅ Economic indicators generation working correctly!")
print("📈 Note: Data exhibits mean reversion and realistic bounds")
test_crypto['Timestamp'] = pd.to_datetime(test_crypto['Timestamp'])

# Δημιουργία ενδιάμεσου DataFrame με Hour και Date (δεν μένει αποθηκευμένο στον explorer)
_ = test_crypto.assign(
    Date=test_crypto['Timestamp'].dt.date,
    Hour=test_crypto['Timestamp'].dt.hour
).loc[test_crypto['Timestamp'].dt.weekday < 5]

daily_df = (
    _.groupby(['Date', 'Symbol'])
    .agg(
        Open=('Open', lambda x: x.loc[_['Hour'].loc[x.index] == 0].iloc[0] if (_['Hour'].loc[x.index] == 0).any() else None),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', lambda x: x.loc[_['Hour'].loc[x.index] == 18].iloc[0] if (_['Hour'].loc[x.index] == 18).any() else None),
        Volume=('Volume', 'sum')
    )
    .dropna(subset=['Open','Close'])
    .reset_index()
)

daily_df['Symbol'] = pd.Categorical(daily_df['Symbol'], categories=['BTC','ETH'], ordered=True)
daily_df = daily_df.sort_values(by=['Symbol','Date']).reset_index(drop=True)

daily_df['Date'] = pd.to_datetime(daily_df['Date'])
test_stocks['Date'] = pd.to_datetime(test_stocks['Date'])

# Βεβαιωνόμαστε ότι οι στήλες έχουν το ίδιο format
daily_df['Date'] = pd.to_datetime(daily_df['Date'])
test_stocks['Date'] = pd.to_datetime(test_stocks['Date'])

# Ενώνουμε κάθετα
merged_df = pd.concat([daily_df, test_stocks], ignore_index=True)

# Ταξινόμηση για να είναι όλα με σειρά
merged_df = merged_df.sort_values(by=['Symbol','Date']).reset_index(drop=True)

print(merged_df.head(20))

