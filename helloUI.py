from flask import Flask, flash, redirect, request
from flask import render_template
from searchuserstry import findsimilartc
from forms import LoginForm
from flask import jsonify
from main import acptcrtlst, clean_text
from tfsconnect import getacptnccriteria

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'


# @page.route('/')
# def home():
#     return render_template('page/home.html')
# app.register_blueprint(page)
@app.route('/')
def hello_world():
    id = 0
    tc_ids = findsimilartc(id)
    return render_template('home.html', tc_ids=tc_ids)


@app.route('/login')
def login():
    form = LoginForm()
    tc_ids = findsimilartc(str(USERSTRYID))
    # return render_template('homechart.html', title='Sign In', form=form, tc_ids=tc_ids)
    return render_template('home.html', title='Sign In', form=form, tc_ids=tc_ids)


@app.route('/search', methods=['GET', 'POST'])
def search():
    form = LoginForm()
    #form.username.data= 7536970
    if request.method == 'POST':
        acptcrt = clean_text(getacptnccriteria(str(form.username.data))).replace("given", "").replace("when","").replace("user", "")
        result = findsimilartc(acptcrt)
        tc_ids = result[0]
        accuracy = result[1]
        to_be_rendered = list(zip(tc_ids, accuracy))
    if request.method == 'GET':
        acptcrt = clean_text(getacptnccriteria(str(USERSTRYID))).replace("given", "").replace("when","").replace("user", "")
        result = findsimilartc(acptcrt)
        tc_ids = result[0]
        accuracy = result[1]
        to_be_rendered = list(zip(tc_ids,accuracy))
    # if form.validate_on_submit():
    #     acptcrt = clean_text(getacptnccriteria(str(form.username.data))).replace("given", "").replace("when","").replace("user", "")



    # return render_template('homechart.html', title='Sign In', form=form, tc_ids=tc_ids)
    #return render_template('home.html', title='Sign In', form=form, tc_ids=tc_ids, accuracy=accuracy)
    return render_template('home.html', title='Sign In', form=form, result =to_be_rendered)





if __name__ == '__main__':
    app.run(debug=True)
