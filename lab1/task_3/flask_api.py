import flask
import pkg_resources
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port")
args = parser.parse_args()

app = flask.Flask(__name__)

ENV_VAR = os.getenv('ENV_VAR', 'Not found!')

@app.route('/', methods=['GET'])
def home():
    return ''.join([
        '<p>This was downloaded from flask service!</p><br>',
        '<p>Environment variable: ',
        ENV_VAR,
        '</p><br><p>Flask version: ',
        pkg_resources.get_distribution('flask').version,
        '</p>'
    ])

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=args.port)