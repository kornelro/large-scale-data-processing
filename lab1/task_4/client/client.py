import requests
import time

done = False

while not done:
    try:
        r = requests.get('http://server:5000')
        print(r.text)
        done = True
    except Exception as e:
        print(e)
        print('Cannot connect to server...')
        time.sleep(3)
