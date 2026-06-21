from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

import json
import os
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

from .models import SensorReading

# ============================================================
#  CONFIGURATION
# ============================================================

TELEGRAM_TOKEN    = '8784024411:AAGt_49V_x5cD5zacnTKBGSkKQIuBpeIIcI'
TELEGRAM_CHAT_ID  = '604412691'
FLOOD_THRESHOLD   = 150.0
WARNING_THRESHOLD = 100.0

# Default dummy data shown when no ESP32 data exists
# This is NOT saved to database — only used for display
DEFAULT_SENSOR_DATA = {
    'temperature':    27.5,
    'humidity':       82.0,
    'water_depth':    65.0,
    'rain_volume':    0.0,
    'wind_speed':     1.5,
    'wind_direction': 'N',
    'flood_risk':     'safe',
    'flood_probability': 12.0,
    'is_dummy': True,   # flag so template knows this is dummy data
}

# TFT model config
MAX_ENCODER_LENGTH   = 24
MAX_PREDICTION_LENGTH = 5

# Load TFT model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR   = os.path.join(BASE_DIR, 'ml')

tft_model   = None
tft_dataset = None

try:
    import torch
    from pytorch_forecasting import TemporalFusionTransformer

    ckpt_path     = os.path.join(ML_DIR, 'tft_flood_model.ckpt')
    dataset_path  = os.path.join(ML_DIR, 'tft_training_dataset.pkl')

    if os.path.exists(ckpt_path) and os.path.exists(dataset_path):
        tft_model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        tft_model.eval()
        with open(dataset_path, 'rb') as f:
            tft_dataset = pickle.load(f)
        print('✅ TFT model loaded successfully!')
    else:
        print('⚠️  TFT model files not found in forecast/ml/')
        print('   Expected: tft_flood_model.ckpt, tft_training_dataset.pkl')

except ImportError:
    print('⚠️  pytorch-forecasting not installed. Run: pip install pytorch-forecasting')
except Exception as e:
    print(f'⚠️  Could not load TFT model: {e}')


# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def send_telegram_alert(message):
    """Send Telegram flood alert."""
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, data={
            'chat_id':    TELEGRAM_CHAT_ID,
            'text':       message,
            'parse_mode': 'HTML'
        }, timeout=5)
    except Exception as e:
        print(f'Telegram error: {e}')


def get_flood_status(water_depth):
    """Return flood status string."""
    if water_depth >= FLOOD_THRESHOLD:
        return 'flood'
    elif water_depth >= WARNING_THRESHOLD:
        return 'warning'
    return 'safe'


def predict_next_5_tft(readings_qs):
    """
    Predict next 5 water depth readings using TFT model.
    Falls back to simple linear extrapolation if TFT unavailable.
    """
    if tft_model is None or tft_dataset is None:
        return predict_next_5_fallback(readings_qs)

    try:
        # Build dataframe from recent readings
        records = list(readings_qs.order_by('-timestamp')[:MAX_ENCODER_LENGTH])
        records = list(reversed(records))

        if len(records) < 6:
            return predict_next_5_fallback(readings_qs)

        df = pd.DataFrame([{
            'time':        r.timestamp,
            'Temperature': r.temperature,
            'Humidity':    r.humidity,
            'Water Depth': r.water_depth,
            'Rain Volume': r.rain_volume,
        } for r in records])

        df['time']      = pd.to_datetime(df['time'], utc=True)
        df['time_idx']  = range(len(df))
        df['hour']      = df['time'].dt.hour
        df['dayofweek'] = df['time'].dt.dayofweek
        df['day']       = df['time'].dt.day
        df['month']     = df['time'].dt.month
        df['series_id'] = '0'

        from pytorch_forecasting import TimeSeriesDataSet

        pred_dataset = TimeSeriesDataSet.from_dataset(
            tft_dataset, df, predict=True, stop_randomization=True
        )
        loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        import torch
        with torch.no_grad():
            raw_preds = tft_model.predict(loader, mode='quantiles')

        median = raw_preds[0, :MAX_PREDICTION_LENGTH, 3].numpy()
        lower  = raw_preds[0, :MAX_PREDICTION_LENGTH, 0].numpy()
        upper  = raw_preds[0, :MAX_PREDICTION_LENGTH, 6].numpy()

        last_time = records[-1].timestamp
        predictions = []

        for i in range(MAX_PREDICTION_LENGTH):
            depth      = float(median[i])
            flood_prob = min(99, max(1, (depth - 45) / 3.0))
            next_time  = last_time + timedelta(minutes=38 * (i + 1))

            predictions.append({
                'time':         next_time.strftime('%H:%M'),
                'water_depth':  round(depth, 1),
                'lower':        round(float(lower[i]), 1),
                'upper':        round(float(upper[i]), 1),
                'flood_prob':   round(flood_prob, 1),
                'flood_status': get_flood_status(depth),
            })

        return predictions

    except Exception as e:
        print(f'TFT prediction error: {e}')
        return predict_next_5_fallback(readings_qs)


def predict_next_5_fallback(readings_qs):
    """
    Simple fallback prediction when TFT is unavailable.
    Uses linear trend from last 6 readings.
    """
    records = list(readings_qs.order_by('-timestamp')[:6])
    if not records:
        return []

    records = list(reversed(records))
    depths  = [r.water_depth for r in records]
    trend   = (depths[-1] - depths[0]) / max(len(depths) - 1, 1)
    last_depth = depths[-1]
    last_time  = records[-1].timestamp
    predictions = []

    for i in range(1, 6):
        depth      = max(0, last_depth + trend * i)
        flood_prob = min(99, max(1, (depth - 45) / 3.0))
        next_time  = last_time + timedelta(minutes=38 * i)

        predictions.append({
            'time':         next_time.strftime('%H:%M'),
            'water_depth':  round(depth, 1),
            'lower':        round(max(0, depth - 5), 1),
            'upper':        round(depth + 5, 1),
            'flood_prob':   round(flood_prob, 1),
            'flood_status': get_flood_status(depth),
        })

    return predictions


def get_dummy_predictions():
    """Default predictions shown when no ESP32 data exists."""
    from datetime import datetime
    import pytz
    now = datetime.now(pytz.timezone('Asia/Kuala_Lumpur'))
    predictions = []
    base_depth = 65.0
    for i in range(1, 6):
        depth     = base_depth + np.random.normal(0, 1)
        next_time = now + timedelta(minutes=38 * i)
        predictions.append({
            'time':         next_time.strftime('%H:%M'),
            'water_depth':  round(depth, 1),
            'lower':        round(depth - 3, 1),
            'upper':        round(depth + 3, 1),
            'flood_prob':   12.0,
            'flood_status': 'safe',
        })
    return predictions


# ============================================================
#  PAGE 1: DASHBOARD
# ============================================================

def dashboard(request):
    """Main dashboard — live sensor data + TFT predictions."""
    latest = SensorReading.objects.order_by('-timestamp').first()

    if latest:
        # Real ESP32 data available
        predictions  = predict_next_5_tft(SensorReading.objects)
        flood_status = get_flood_status(latest.water_depth)
        recent       = SensorReading.objects.order_by('-timestamp')[:10]

        context = {
            'latest':          latest,
            'flood_status':    flood_status,
            'predictions':     predictions,
            'recent_readings': recent,
            'last_updated':    latest.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'is_dummy':        False,
            'model_name':      'TFT' if tft_model else 'Linear Extrapolation',
            'pred1': predictions[0] if len(predictions) > 0 else None,
            'pred2': predictions[1] if len(predictions) > 1 else None,
            'pred3': predictions[2] if len(predictions) > 2 else None,
            'pred4': predictions[3] if len(predictions) > 3 else None,
            'pred5': predictions[4] if len(predictions) > 4 else None,
        }
    else:
        # No ESP32 data — show default dummy data
        predictions = get_dummy_predictions()
        context = {
            'latest':          None,
            'dummy':           DEFAULT_SENSOR_DATA,
            'flood_status':    'safe',
            'predictions':     predictions,
            'recent_readings': [],
            'last_updated':    'Waiting for ESP32 data...',
            'is_dummy':        True,
            'model_name':      'TFT' if tft_model else 'Fallback',
            'pred1': predictions[0] if len(predictions) > 0 else None,
            'pred2': predictions[1] if len(predictions) > 1 else None,
            'pred3': predictions[2] if len(predictions) > 2 else None,
            'pred4': predictions[3] if len(predictions) > 3 else None,
            'pred5': predictions[4] if len(predictions) > 4 else None,
        }

    return render(request, 'dashboard.html', context)


# ============================================================
#  PAGE 2: HISTORY
# ============================================================

def history(request):
    """History page with calendar navigation."""
    dates_with_data = SensorReading.objects.dates('timestamp', 'day')
    date_list = [d.strftime('%Y-%m-%d') for d in dates_with_data]
    return render(request, 'history.html', {
        'dates_with_data': json.dumps(date_list),
    })


def get_history_data(request):
    """API — returns sensor readings for a selected date."""
    date_str = request.GET.get('date', '')
    if not date_str:
        return JsonResponse({'error': 'No date provided'}, status=400)

    try:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return JsonResponse({'error': 'Invalid date format'}, status=400)

    readings = SensorReading.objects.filter(
        timestamp__date=selected_date
    ).order_by('timestamp')

    if not readings.exists():
        return JsonResponse({'date': date_str, 'count': 0, 'readings': [],
                             'message': 'No data found for this date'})

    data = [{
        'timestamp':         r.timestamp.strftime('%H:%M:%S'),
        'temperature':       round(r.temperature, 2),
        'humidity':          round(r.humidity, 2),
        'water_depth':       round(r.water_depth, 2),
        'rain_volume':       round(r.rain_volume, 2),
        'wind_speed':        round(r.wind_speed, 2),
        'wind_direction':    r.wind_direction,
        'flood_status':      r.flood_risk,
        'flood_probability': round(r.flood_probability, 1),
    } for r in readings]

    depths  = [r.water_depth for r in readings]
    summary = {
        'max_depth':    round(max(depths), 2),
        'min_depth':    round(min(depths), 2),
        'avg_depth':    round(np.mean(depths), 2),
        'flood_events': sum(1 for r in readings if r.flood_risk == 'flood'),
        'total_rain':   round(sum(r.rain_volume for r in readings), 2),
    }

    return JsonResponse({'date': date_str, 'count': readings.count(),
                         'summary': summary, 'readings': data})


# ============================================================
#  PAGE 3: ABOUT
# ============================================================

def about(request):
    """About page."""
    context = {
        'telegram_bot_url': 'https://t.me/FloodForecastMark1Bot',
        'email':            'wmhzq02@gmail.com',
        'linkedin':         'https://www.linkedin.com/in/w-m-haziq-138155321',
        'project_name':     'River Flood Forecasting Device',
        'bot_name':         'Flood Forecast Mark 1',
        'model_name':       'Temporal Fusion Transformer (TFT)',
    }
    return render(request, 'about.html', context)


# ============================================================
#  ESP32 API ENDPOINT
# ============================================================

@csrf_exempt
def receive_sensor_data(request):
    """
    ESP32 sends sensor data here via HTTP POST.

    Expected JSON:
    {
        "temperature":   26.5,
        "humidity":      85.2,
        "water_depth":   67.4,
        "rain_volume":   2.1,
        "wind_speed":    3.2,
        "wind_direction": 180.0
    }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        body           = json.loads(request.body)
        temperature    = float(body.get('temperature', 0))
        humidity       = float(body.get('humidity', 0))
        water_depth    = float(body.get('water_depth', 0))
        rain_volume    = float(body.get('rain_volume', 0))
        wind_speed     = float(body.get('wind_speed', 0))
        wind_direction = str(body.get('wind_direction', 'N'))
        flood_status   = get_flood_status(water_depth)

        # Save to database
        reading = SensorReading.objects.create(
            temperature       = temperature,
            humidity          = humidity,
            water_depth       = water_depth,
            rain_volume       = rain_volume,
            wind_speed        = wind_speed,
            wind_direction    = wind_direction,
            flood_risk        = flood_status,
            flood_probability = 0.0,
        )

        # TFT flood probability prediction
        flood_probability = min(99, max(1, (water_depth - 45) / 3.0))
        try:
            predictions = predict_next_5_tft(SensorReading.objects)
            if predictions:
                flood_probability = predictions[0]['flood_prob']
        except Exception as e:
            print(f'Prediction error: {e}')

        reading.flood_probability = round(flood_probability, 1)
        reading.save()

        # Telegram alerts
        if flood_status == 'flood':
            send_telegram_alert(
                f'🚨 <b>FLOOD ALERT!</b>\n\n'
                f'📍 River Flood Forecasting Device\n'
                f'💧 Water Depth: <b>{water_depth}cm</b> ‼️ DANGER\n'
                f'🌡️ Temperature: {temperature}°C\n'
                f'💦 Humidity: {humidity}%\n'
                f'🌧️ Rain: {rain_volume}mm\n'
                f'📊 Flood Probability: <b>{round(flood_probability, 1)}%</b>\n'
                f'🕐 {reading.timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
            )
        elif flood_status == 'warning':
            send_telegram_alert(
                f'⚠️ <b>FLOOD WARNING!</b>\n\n'
                f'📍 River Flood Forecasting Device\n'
                f'💧 Water Depth: <b>{water_depth}cm</b> ⚠️ WARNING\n'
                f'🌡️ Temperature: {temperature}°C\n'
                f'💦 Humidity: {humidity}%\n'
                f'🌧️ Rain: {rain_volume}mm\n'
                f'📊 Flood Probability: <b>{round(flood_probability, 1)}%</b>\n'
                f'🕐 {reading.timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
            )

        return JsonResponse({
            'status':            'success',
            'flood_status':      flood_status,
            'flood_probability': round(flood_probability, 1),
            'reading_id':        reading.id,
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
