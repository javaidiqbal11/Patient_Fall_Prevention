from flask import Flask, request, jsonify

# Flask APP used to deploy for the testing 

app = Flask(__name__)


@app.route('/fallalert', methods=['GET', 'POST'])
def hello():
    print("In the endpoint---")
    data = request.get_data(as_text=True)
    print("Data : {}".format(data))
    return jsonify({"data": data})


if __name__ == "__main__":
    app.run(debug=True, port=5555)
