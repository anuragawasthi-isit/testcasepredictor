from flask import Flask
from flask import render_template
from searchuserstry import findsimilartc, findsimilartcwitacc
from flask import jsonify
from main import acptcrtlst, clean_text
from tfsconnect import getacptnccriteria

app=Flask(__name__)

# @page.route('/')
# def home():
#     return render_template('page/home.html')
#app.register_blueprint(page)
@app.route('/<id>')
def hello_world(id):
    id = int(id)
    acptcrt =clean_text(getacptnccriteria(id)).replace("give","").replace("when","").replace("user","")
    return jsonify({"test_case_ids": findsimilartcwitacc(acptcrt)})


if __name__ == '__main__':
   app.run(debug = True)