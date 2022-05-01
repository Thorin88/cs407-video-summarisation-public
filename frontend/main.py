from os import path
from flask import Flask, render_template, request, redirect, send_from_directory
from flask.helpers import send_file
import subprocess
from subprocess import Popen, PIPE

app = Flask(__name__)

multiple = False
tvsum = False
setting = ''

@app.route('/download')
def download():
    return send_file('static/summary.webm')

@app.route('/', methods=['GET', 'POST'])
def home():
    global multiple
    global tvsum
    global setting
    print("multiple:" + str(multiple))
    if request.method == 'POST':
        file = request.files['video']
        if file:
            filename = (file.filename)
            file.save('./uploaded.webm')

        filepath = 'backend/pipeline/models/ab_basic/checkpoint/'
        if (tvsum):
            filepath += 'tvsum'
        else:
            filepath += 'summe'
        filepath += setting + '.yml.0.pt'
        if multiple: # Slightly different check for multiple models
            filepathToCheck = filepath.replace(".yml","_multi_0")
        else:
            filepathToCheck = filepath
        if (path.exists(filepathToCheck)):
            command = 'python3 backend/pipeline/src/my_infer.py basic --ckpt-path ' + filepath + ' --source ./uploaded.webm --save-path static/summary.webm'
            if (multiple):
                command += ' --pipeline-mode multi'
            subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
            return redirect('download')
        else:
            multiple = False
            tvsum = False
            setting = ''
            return render_template('home.html', value1=filepath + " is missing")

    return render_template('home.html')


@app.route('/pipeline-multiple', methods=['POST'])
def pipelineMultiple():
    global multiple
    multiple = True
    return render_template('home.html')

@app.route('/pipeline-single', methods=['POST'])
def pipelineSingle():
    global multiple
    multiple = False
    return render_template('home.html')

@app.route('/dataset-tvsum', methods=['POST'])
def datasetTvSum():
    global tvsum
    tvsum = True
    return render_template('home.html')

@app.route('/dataset-summe', methods=['POST'])
def datasetSumMe():
    global tvsum
    tvsum = False
    return render_template('home.html')

@app.route('/setting-canonical', methods=['POST'])
def settingCanonical():
    global setting
    setting = ''
    return render_template('home.html')

@app.route('/setting-augmented', methods=['POST'])
def settingAugmented():
    global setting
    setting = '_aug'
    return render_template('home.html')

@app.route('/setting-transfer', methods=['POST'])
def settingTransfer():
    global setting
    setting = '_trans'
    return render_template('home.html')

app.run(debug=True, port=5000)
