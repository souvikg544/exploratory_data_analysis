from flask import Flask , render_template, request, redirect, url_for, flash, jsonify
import numpy as np
from model import predict

app = Flask(__name__)



@app.route('/')

def home():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def guess():
    if request.method == 'POST':
        text = request.form['text']
        res = predict(text)
        return render_template('index.html',res=res)
    
if __name__=='__main__':
    app.run(debug=True)

