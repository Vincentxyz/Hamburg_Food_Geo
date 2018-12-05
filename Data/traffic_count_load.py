import pandas as pd

tf = pd.read_json('traffic_count.json')

tags = tf(['tags'])