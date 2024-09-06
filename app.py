from routes.qna_route import qna_function
from flask import Flask

app = Flask(__name__)

@app.route('/qna', methods=['POST'])
def qna():
    return qna_function()

if __name__ == '__main__':
    app.run(debug=True)