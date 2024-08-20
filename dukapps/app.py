from flask import Flask, render_template, request
import os


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('project/public/index.html')

if __name__ == '__main__':
    app.run(debug=True)
