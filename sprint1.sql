WITH crypto_daily AS (
    SELECT
        DATE("timestamp") AS day,
        symbol,
        'crypto' AS asset_type,
        FIRST_VALUE(open) OVER (
            PARTITION BY symbol, DATE("timestamp") 
            ORDER BY "timestamp" ASC
        ) AS open,
        FIRST_VALUE(close) OVER (
            PARTITION BY symbol, DATE("timestamp") 
            ORDER BY "timestamp" DESC
        ) AS close,
        MAX(high) OVER (PARTITION BY symbol, DATE("timestamp")) AS high,
        MIN(low) OVER (PARTITION BY symbol, DATE("timestamp")) AS low,
        SUM(volume) OVER (PARTITION BY symbol, DATE("timestamp")) AS volume
    FROM crypto_prices
    WHERE symbol IN ('BTC','ETH')
      AND "timestamp" BETWEEN '2020-01-01' AND '2024-12-31'
),
stock_daily AS (
    SELECT
        date AS day,
        symbol,
        'stock' AS asset_type,
        open,
        high,
        low,
        close,
        volume
    FROM stock_prices
    WHERE symbol IN ('MSFT','PG','JNJ','V','COST')
      AND date BETWEEN '2020-01-01' AND '2024-12-31'
)
SELECT DISTINCT day, symbol, asset_type, open, high, low, close, volume
FROM (
    SELECT * FROM crypto_daily
    UNION ALL
    SELECT * FROM stock_daily
) t
WHERE EXTRACT(DOW FROM day) BETWEEN 1 AND 5
ORDER BY day DESC, symbol ASC;


