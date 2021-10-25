from flask import Flask, request
from flask import render_template
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def hello_world():
    return "Welcome to the world Mr. Great Project!"

@app.route("/hello.html")
def hello():
    return render_template("hello.Html", var1="cool var 1")

@app.route("/test_static.html")
def test_static():
    return app.send_static_file("test.pdf")

@app.route("/upload_scene", methods=["POST"])
@cross_origin()
def upload_scene():
    data = request.form 
    print(data)
    print(request.json)
    with open("updated_scene.json", "w") as f:
        json.dump(request.json, f)
    return "Scene successfully uploaded!"

@app.route("/refresh_scene", methods=["POST"])
@cross_origin()
def refresh_scene():
    data = request.form 
    with open("refresh_scene.json", "w") as f:
        json.dump(request.json, f)
    return "Scene refresh request received!"