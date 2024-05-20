from difflib import *
import os
import urllib.request
from db import *
from flask import *
import sqlite3
import string
import re
import speech_recognition as sr
import nltk
from profanityfilter import ProfanityFilter
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from checkcomment import *
# from checkspam import *
app = Flask(__name__)
app.secret_key = "secret key"
language = 'en'
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.secret_key = "secret key"
app.config['ALLOWED_EXTENSIONS'] = set(['mp3', 'wav'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def predict_text(text):
    try:
        from keras.preprocessing.text import one_hot
        from keras.utils import pad_sequences
        import re
        from nltk.stem.snowball import SnowballStemmer
        from nltk.corpus import stopwords
        from tensorflow.keras.models import load_model
        import numpy as np

        # Load the saved model
        model = load_model("one.h5")
        # Text cleaning
        text_cleaning = "\b0\S*|\b[^A-Za-z0-9]+"

        stop_words = stopwords.words('english')

        def preprocess_filter(text, stem=False):
            text = re.sub(text_cleaning, " ", str(text.lower()).strip())
            tokens = []
            for token in text.split():
                if token not in stop_words:
                    if stem:
                        stemmer = SnowballStemmer(language='english')
                        token = stemmer.stem(token)
                    tokens.append(token)
            return " ".join(tokens)

        def one_hot_encoded(text, vocab_size=5000, max_length=40):
            hot_encoded = one_hot(text, vocab_size)
            return hot_encoded

            # word embedding pipeline

        def word_embedding(text):
            preprocessed_text = preprocess_filter(text)
            return one_hot_encoded(preprocessed_text)

        # Define the function for prediction input processing

        def prediction_input_processing(text):
            try:
                max_length = 1000
                encoded = word_embedding(text)
                padded_encoded_title = pad_sequences(
                    [encoded], maxlen=max_length, padding='pre')
                output = model.predict(padded_encoded_title)
                output = np.where(0.4 > output, 1, 0)
                if output[0][0] == 1:
                    return 'yes'
                return 'no'
            except:
                pass
        prediction_input_processing(text)
    except:
        pass


def checkval(a):
    e = []
    ps = nltk.WordNetLemmatizer()
    ae = nltk.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words("english")
    text = "".join([word for word in a if word not in string.punctuation])
    text = [word for word in text.split() if word not in stopwords]
    text = [ps.lemmatize(word) for word in text]
    text = " ".join([ae.stem(word) for word in text])
    return text


@app.route('/block', methods=['GET', 'POST'])
def block():
    vu = open("block.txt", "r").read()
    da = json.loads(vu)
    x = [str(x)for x in da]
    x = "**".join(x)
    return x


@app.route("/front")
def front():
    return render_template("front.html")


@app.route("/")
def data():
    return render_template("page1.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/boorish")
def index():
    return render_template("boorish.html")


@app.route("/nexts")
def nexts():
    return render_template("nlp.html")


@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")


@app.route('/logon')
def logon():
    return render_template('signing.html')


@app.route('/take')
def take():
    return render_template('spamlink.html')


@app.route('/vu')
def vu():
    return render_template('student.html')


@app.route("/checkcomment", methods=["post"])
def check():
    data = request.form['data']
    data = checkval(data)
    predict_text(data)
    print(data)
    pf = ProfanityFilter()
    Approval = pf.is_profane(data)
    if Approval:
        v = "yes"
    else:
        v = "no"
    return v


@app.route('/spam', methods=['GET', 'POST'])
def spam():
    vu = open("spamlink.txt", "r").read()
    return vu


@app.route('/nonspam', methods=['GET', 'POST'])
def nonspam():
    vu = open("nonspamlink", "r").read()
    return vu


@app.route('/wrong', methods=['GET', 'POST'])
def wrong():
    vu = open("wrong", "r").read()
    return vu


@app.route('/vul', methods=['GET', 'POST'])
def vul():
    vu = open("vulgar.txt", "r").read()
    return vu


@app.route('/nonvul', methods=['GET', 'POST'])
def nonvul():
    vu = open("nonvulgar.txt", "r").read()
    return vu


@app.route('/next2')
def next2():
    return render_template("count2.html")


@app.route('/next')
def next():
    return render_template("count.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        dataset = panda.read_csv(f.filename)
        tweet = dataset.username
        user = dataset.tweet
        x = user.to_list()
        print(x)
        vulgarwordtweet = []
        nonvulgarword = []
        for k in x:
            try:
                data = checkval(k)
                print(data)
                pf = ProfanityFilter()
                Approval = pf.is_profane(data)
                if Approval:
                    vulgarwordtweet.append(data)
                else:
                    nonvulgarword.append(data)
            except:
                pass
        x = json.dumps(vulgarwordtweet)
        x2 = json.dumps(nonvulgarword)

        vu = open("vulgar.txt", "w")
        vu.write(x)
        vu.close()
        vu = open("nonvulgar.txt", "w")
        vu.write(x2)
        vu.close()

        session["value"] = str(len(vulgarwordtweet)) + \
            "**"+str(len(nonvulgarword))

        return redirect("next")


@app.route("/signup", methods=["post"])
def signup():
    username = request.form['user']
    name = request.form['name']
    email = request.form['email']
    number = request.form["mobile"]
    password = request.form['password']
    role = "student"
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`,'role') VALUES (?, ?, ?, ?, ?,?)",
                (username, email, password, number, name, role))
    con.commit()
    con.close()
    return render_template("page1.html")


@app.route("/signin", methods=["post"])
def signin():

    mail1 = request.form['user']
    password1 = request.form['password']
    print(mail1, password1)
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute(
        "select `user`, `password`,role from info where `user` = ? AND `password` = ?", (mail1, password1,))
    data = cur.fetchone()

    if mail1 == 'admin' and password1 == 'admin':
        session['username'] = "admin"
        return render_template("student.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        session['username'] = data[0]
        return render_template("student.html")
    else:
        return render_template("signup.html")


@app.route('/logout')
def home():
    session.pop('username', None)
    return redirect("/")


def translate_tamil_to_english(text):
    from googletrans import Translator
    translator = Translator()
    translated_text = translator.translate(text, src='ta', dest='en')
    return translated_text.text


@app.route('/checkvoice', methods=['POST'])
def checkvoice():
    f = request.files['x']
    data = translate_tamil_to_english(f)

    pf = ProfanityFilter()
    Approval = pf.is_profane(data)
    if Approval:
        v = "yes"
    else:
        v = "no"
    return v


@app.route('/uploadajax', methods=['POST'])
def upldfile():
    dir = os.path.join(basedir, 'upload/')
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    if request.method == 'POST':
        files = request.files['file']
        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            app.logger.info('FileName: ' + filename)
            updir = os.path.join(basedir, 'upload/')
            files.save(os.path.join(updir, filename))
            file_size = os.path.getsize(os.path.join(updir, filename))
            print(file_size)
            return jsonify(name=filename, size=file_size)


@app.route('/translate', methods=['POST'])
def translate():
    from googletrans import Translator
    translator = Translator()
    data = request.get_json()
    tamil_text = data['tamil_text']
    english_text = translator.translate(tamil_text, src='ta', dest='en').text
    pf = ProfanityFilter()
    predict_text(english_text)
    Approval = pf.is_profane(english_text)
    if Approval:
        v = "yes"
    else:
        v = "no"

    return jsonify({'english_text': english_text, "value": v})


if __name__ == '__main__':
    app.run()
