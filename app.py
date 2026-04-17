from flask import Flask, render_template, request, jsonify
from ai_model import run_ai

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    product = request.form['product']
    platform = request.form['platform']
    impressions = int(request.form['impressions'])
    clicks = int(request.form['clicks'])

    ai_result = run_ai(product, platform, impressions, clicks)

    return render_template('result.html', **ai_result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json

    result = run_ai(
        data['product'],
        data['platform'],
        data['impressions'],
        data['clicks']
    )

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)