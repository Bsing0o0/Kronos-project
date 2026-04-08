import os
import pandas as pd
import numpy as np
import json
import re
import csv
import urllib.request
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}


def infer_symbol_from_path(file_path):
    """Infer a stock symbol from the filename for display."""
    if not file_path:
        return 'Unknown'

    base_name = os.path.basename(file_path)
    name_no_ext = os.path.splitext(base_name)[0]

    # Common auto-generated pattern from this project: RY_TO_5m_7d -> RY.TO
    parts = name_no_ext.split('_')
    if len(parts) >= 2 and len(parts[0]) <= 8 and len(parts[1]) <= 4:
        exch = parts[1].upper()
        if exch.isalpha():
            return f"{parts[0].upper()}.{exch}"

    return name_no_ext


def fetch_yahoo_chart_to_csv(symbol, interval='5m', range_value='7d'):
    """Fetch Yahoo chart data and save to data/*.csv in Kronos format."""
    query_symbol = symbol.strip().upper()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{query_symbol}?interval={interval}&range={range_value}"
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json,text/plain,*/*',
            'Referer': 'https://finance.yahoo.com/'
        },
    )

    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        raise RuntimeError(f"Yahoo request failed: {str(e)}")

    chart = data.get('chart', {})
    if chart.get('error'):
        raise RuntimeError(f"Yahoo API error: {chart['error']}")

    result_list = chart.get('result') or []
    if not result_list:
        raise RuntimeError("Yahoo API returned no result data")

    result = result_list[0]
    quote = ((result.get('indicators') or {}).get('quote') or [{}])[0]
    timestamps = result.get('timestamp') or []
    opens = quote.get('open') or []
    highs = quote.get('high') or []
    lows = quote.get('low') or []
    closes = quote.get('close') or []
    volumes = quote.get('volume') or []

    rows = []
    for i, ts in enumerate(timestamps):
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        l = lows[i] if i < len(lows) else None
        c = closes[i] if i < len(closes) else None
        v = volumes[i] if i < len(volumes) else None

        if None in (o, h, l, c, v):
            continue

        dt_value = datetime.datetime.fromtimestamp(int(ts), datetime.UTC).replace(tzinfo=None)
        amount = float(c) * float(v)
        rows.append([
            dt_value.strftime('%Y-%m-%d %H:%M:%S'),
            float(o),
            float(h),
            float(l),
            float(c),
            float(v),
            float(amount),
        ])

    if len(rows) == 0:
        raise RuntimeError("No valid OHLCV rows returned from Yahoo")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)

    safe_symbol = re.sub(r'[^A-Za-z0-9]+', '_', query_symbol).strip('_')
    out_name = f"{safe_symbol}_{interval}_{range_value}.csv"
    out_path = os.path.join(data_dir, out_name)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        writer.writerows(rows)

    return out_path, len(rows)

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB",
                    'symbol': infer_symbol_from_path(file_path)
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0, future_mode=False):
    """Create prediction chart"""
    historical_up = '#4BD7FF'
    historical_down = '#2F7BFF'
    predicted_up = '#8B5CF6'
    predicted_down = '#EC4899'
    actual_up = '#F59E0B'
    actual_down = '#F97316'

    if future_mode:
        # Future mode: always use the last `lookback` rows as historical context
        historical_df = df.tail(lookback).copy()
    elif historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
    
    # Create chart
    fig = go.Figure()

    def add_gradient_band(x_values, colors, opacities):
        if x_values is None or len(x_values) < 2:
            return

        segments = len(colors)
        for index, color in enumerate(colors):
            start_idx = int(index * (len(x_values) - 1) / segments)
            end_idx = int((index + 1) * (len(x_values) - 1) / segments)
            if end_idx <= start_idx:
                continue

            fig.add_vrect(
                x0=x_values[start_idx],
                x1=x_values[end_idx],
                fillcolor=color,
                opacity=opacities[index],
                line_width=0,
                layer='below'
            )
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name=f'Historical Data ({len(historical_df)} data points)',
        increasing_line_color=historical_up,
        decreasing_line_color=historical_down,
        increasing_fillcolor='rgba(75, 215, 255, 0.55)',
        decreasing_fillcolor='rgba(47, 123, 255, 0.42)'
    ))

    historical_x = historical_df['timestamps'].tolist() if 'timestamps' in historical_df.columns else list(historical_df.index)
    add_gradient_band(
        historical_x,
        ['rgba(37, 99, 235, 1)', 'rgba(75, 215, 255, 1)', 'rgba(14, 165, 233, 1)'],
        [0.05, 0.08, 0.06]
    )
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name=f'Prediction Data ({len(pred_df)} data points)',
            increasing_line_color=predicted_up,
            decreasing_line_color=predicted_down,
            increasing_fillcolor='rgba(139, 92, 246, 0.52)',
            decreasing_fillcolor='rgba(236, 72, 153, 0.42)'
        ))

        add_gradient_band(
            list(pred_timestamps),
            ['rgba(91, 93, 252, 1)', 'rgba(139, 92, 246, 1)', 'rgba(236, 72, 153, 1)'],
            [0.06, 0.1, 0.08]
        )
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name=f'Actual Data ({len(actual_df)} data points)',
            increasing_line_color=actual_up,
            decreasing_line_color=actual_down,
            increasing_fillcolor='rgba(245, 158, 11, 0.45)',
            decreasing_fillcolor='rgba(249, 115, 22, 0.35)'
        ))
    
    # Update layout
    chart_title = (
        f'Kronos Future Forecast — {lookback} Historical + {pred_len} Predicted Candles'
        if future_mode else
        f'Kronos Backtest — {lookback} Historical vs {pred_len} Predicted + Actual'
    )
    fig.update_layout(
        title=chart_title,
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_dark',
        height=600,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(3, 17, 31, 0.18)',
        legend=dict(
            bgcolor='rgba(7, 22, 38, 0.72)',
            bordercolor='rgba(75, 215, 255, 0.18)',
            borderwidth=1,
            font=dict(color='#DDF5FF')
        ),
        font=dict(color='#DDF5FF')
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'symbol': infer_symbol_from_path(file_path),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500


def _INTERVAL_TO_AV(interval):
    """Map UI interval string to Alpha Vantage interval."""
    mapping = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '60min', '60m': '60min'}
    return mapping.get(interval, '5min')


def _INTERVAL_TO_TD(interval):
    """Map UI interval string to Twelve Data interval."""
    mapping = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '60m': '1h', '1d': '1day', '5d': '1week'}
    return mapping.get(interval, '5min')


def fetch_alpha_vantage_to_csv(symbol, interval, api_key):
    """Fetch intraday bars from Alpha Vantage and save to data/*.csv."""
    av_interval = _INTERVAL_TO_AV(interval)
    query_symbol = symbol.strip().upper()
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={query_symbol}&interval={av_interval}&outputsize=full&apikey={api_key}"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        raise RuntimeError(f"Alpha Vantage request failed: {e}")

    if 'Information' in data:
        raise RuntimeError(f"Alpha Vantage limit/error: {data['Information']}")
    if 'Error Message' in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")

    ts_key = f"Time Series ({av_interval})"
    ts = data.get(ts_key)
    if not ts:
        raise RuntimeError(f"Alpha Vantage returned no time series data (key expected: '{ts_key}')")

    rows = []
    for dt_str, ohlcv in sorted(ts.items()):
        try:
            o = float(ohlcv['1. open'])
            h = float(ohlcv['2. high'])
            l = float(ohlcv['3. low'])
            c = float(ohlcv['4. close'])
            v = float(ohlcv['5. volume'])
            rows.append([dt_str, o, h, l, c, v, c * v])
        except (KeyError, ValueError):
            continue

    if not rows:
        raise RuntimeError("No valid OHLCV rows from Alpha Vantage")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    safe_sym = re.sub(r'[^A-Za-z0-9]+', '_', query_symbol).strip('_')
    out_path = os.path.join(data_dir, f"{safe_sym}_{av_interval}_av.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        writer.writerows(rows)
    return out_path, len(rows)


def fetch_twelve_data_to_csv(symbol, interval, api_key):
    """Fetch bars from Twelve Data and save to data/*.csv."""
    td_interval = _INTERVAL_TO_TD(interval)
    query_symbol = symbol.strip().upper()
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={query_symbol}&interval={td_interval}&outputsize=5000&apikey={api_key}"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        raise RuntimeError(f"Twelve Data request failed: {e}")

    if data.get('status') == 'error':
        raise RuntimeError(f"Twelve Data error: {data.get('message', 'unknown error')}")

    values = data.get('values')
    if not values:
        raise RuntimeError("Twelve Data returned no values")

    rows = []
    for bar in reversed(values):
        try:
            o = float(bar['open'])
            h = float(bar['high'])
            l = float(bar['low'])
            c = float(bar['close'])
            v = float(bar.get('volume', 0))
            rows.append([bar['datetime'], o, h, l, c, v, c * v])
        except (KeyError, ValueError):
            continue

    if not rows:
        raise RuntimeError("No valid OHLCV rows from Twelve Data")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    safe_sym = re.sub(r'[^A-Za-z0-9]+', '_', query_symbol).strip('_')
    out_path = os.path.join(data_dir, f"{safe_sym}_{td_interval}_td.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        writer.writerows(rows)
    return out_path, len(rows)


@app.route('/api/fetch-yahoo', methods=['POST'])
def fetch_yahoo_data():
    """Fetch latest market bars from Yahoo and save as a local CSV."""
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or '').strip()
        interval = (data.get('interval') or '5m').strip()
        range_value = (data.get('range') or '7d').strip()

        if not symbol:
            return jsonify({'error': 'Symbol cannot be empty'}), 400

        out_path, row_count = fetch_yahoo_chart_to_csv(symbol, interval=interval, range_value=range_value)

        return jsonify({
            'success': True,
            'file_path': out_path,
            'rows': row_count,
            'message': f'Fetched {row_count} rows for {symbol} and saved to data folder'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch Yahoo data: {str(e)}'}), 500


@app.route('/api/api-keys', methods=['GET'])
def get_api_keys():
    """Return saved API keys (for pre-filling UI fields)."""
    keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_keys.json')
    if os.path.exists(keys_path):
        try:
            with open(keys_path, 'r') as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({'alphavantage': '', 'twelvedata': ''})


@app.route('/api/fetch-market', methods=['POST'])
def fetch_market_data():
    """Unified market data fetch endpoint (Yahoo only)."""
    try:
        data = request.get_json() or {}
        source = (data.get('source') or 'yahoo').strip().lower()
        symbol = (data.get('symbol') or '').strip()
        interval = (data.get('interval') or '5m').strip()
        range_value = (data.get('range') or '7d').strip()

        if not symbol:
            return jsonify({'error': 'Symbol cannot be empty'}), 400

        if source != 'yahoo':
            return jsonify({'error': 'Only Yahoo source is supported in this build'}), 400

        out_path, row_count = fetch_yahoo_chart_to_csv(symbol, interval=interval, range_value=range_value)

        return jsonify({
            'success': True,
            'file_path': out_path,
            'rows': row_count,
            'message': f'[{source}] Fetched {row_count} rows for {symbol}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 120))
        pred_len = int(data.get('pred_len', 7))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        predict_mode = data.get('predict_mode', 'future')  # 'future' or 'backtest'

        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400

        if lookback < 20:
            return jsonify({'error': 'lookback must be >= 20'}), 400
        if pred_len < 1:
            return jsonify({'error': 'pred_len must be >= 1'}), 400

        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400

        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400

        # Perform prediction
        if not (MODEL_AVAILABLE and predictor is not None):
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400

        try:
            required_cols = ['open', 'high', 'low', 'close']
            if 'volume' in df.columns:
                required_cols.append('volume')

            window_warning = None

            if predict_mode == 'future':
                # ── FUTURE FORECAST MODE ──────────────────────────────────────
                # Use the last `lookback` rows; timestamps extend beyond last bar.
                x_df = df.tail(lookback)[required_cols].copy()
                x_timestamp = df.tail(lookback)['timestamps'].copy()

                time_diff = (df['timestamps'].iloc[-1] - df['timestamps'].iloc[-2]
                             if len(df) > 1 else pd.Timedelta(hours=1))
                y_timestamp = pd.Series(
                    pd.date_range(start=df['timestamps'].iloc[-1] + time_diff,
                                  periods=pred_len, freq=time_diff),
                    name='timestamps'
                )
                prediction_type = f"Future Forecast — next {pred_len} candles after {df['timestamps'].iloc[-1].strftime('%Y-%m-%d %H:%M')}"
                actual_data = []
                actual_df = None

            else:
                # ── BACKTEST MODE ────────────────────────────────────────────
                start_date = data.get('start_date')
                used_start_date = start_date
                selected_window_df = None

                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]

                    if len(time_range_df) < lookback + pred_len:
                        selected_window_df = df.tail(lookback + pred_len).copy()
                        used_start_date = None
                        window_warning = (
                            f"Selected start had only {len(time_range_df)} points. "
                            f"Auto-shifted to latest {lookback + pred_len}-point window."
                        )
                    else:
                        selected_window_df = time_range_df.iloc[:lookback + pred_len].copy()

                    x_df = selected_window_df.iloc[:lookback][required_cols]
                    x_timestamp = selected_window_df.iloc[:lookback]['timestamps']
                    y_timestamp = selected_window_df.iloc[lookback:lookback + pred_len]['timestamps']
                    time_span = selected_window_df['timestamps'].iloc[lookback + pred_len - 1] - selected_window_df['timestamps'].iloc[0]
                    prediction_type = f"Backtest (window: {lookback} historical + {pred_len} predicted, span: {time_span})"
                else:
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback + pred_len]['timestamps']
                    prediction_type = "Backtest (first window)"

                # Build actual data for comparison
                if selected_window_df is not None and len(selected_window_df) >= lookback + pred_len:
                    actual_df = selected_window_df.iloc[lookback:lookback + pred_len]
                elif len(df) >= lookback + pred_len and not start_date:
                    actual_df = df.iloc[lookback:lookback + pred_len]
                else:
                    actual_df = None

                actual_data = []
                if actual_df is not None:
                    for _, row in actual_df.iterrows():
                        actual_data.append({
                            'timestamp': row['timestamps'].isoformat(),
                            'open': float(row['open']), 'high': float(row['high']),
                            'low': float(row['low']),   'close': float(row['close']),
                            'volume': float(row['volume']) if 'volume' in row else 0,
                            'amount': float(row['amount']) if 'amount' in row else 0
                        })

            # Ensure Series type for timestamps
            if isinstance(x_timestamp, pd.DatetimeIndex):
                x_timestamp = pd.Series(x_timestamp, name='timestamps')
            if isinstance(y_timestamp, pd.DatetimeIndex):
                y_timestamp = pd.Series(y_timestamp, name='timestamps')

            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count
            )

        except Exception as e:
            return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500

        # Build prediction result rows with timestamps
        if predict_mode == 'future':
            pred_timestamps = pd.DatetimeIndex(y_timestamp.values)
        else:
            time_diff = (df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                         if len(df) > 1 else pd.Timedelta(hours=1))
            ref_df = x_df if predict_mode == 'future' else (
                selected_window_df if 'selected_window_df' in dir() and selected_window_df is not None else df.iloc[:lookback])
            last_ts = x_timestamp.iloc[-1]
            pred_timestamps = pd.date_range(start=last_ts + time_diff, periods=pred_len, freq=time_diff)

        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            ts_str = pd.Timestamp(pred_timestamps[i]).isoformat() if i < len(pred_timestamps) else f"T{i}"
            prediction_results.append({
                'timestamp': ts_str,
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']),   'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })

        # Create chart
        if predict_mode == 'future':
            chart_json = create_prediction_chart(df, pred_df, lookback, pred_len,
                                                  actual_df=None, future_mode=True)
        else:
            historical_start_idx = (selected_window_df.index[0]
                                     if 'selected_window_df' in dir() and selected_window_df is not None else 0)
            chart_json = create_prediction_chart(df, pred_df, lookback, pred_len,
                                                  actual_df=actual_df,
                                                  historical_start_idx=historical_start_idx,
                                                  future_mode=False)

        # Save results
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback, 'pred_len': pred_len,
                    'temperature': temperature, 'top_p': top_p,
                    'sample_count': sample_count, 'mode': predict_mode,
                    'window_warning': window_warning
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")

        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else ''),
            'warning': window_warning
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070)
