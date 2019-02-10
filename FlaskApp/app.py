import logging

from flask import Flask, Markup, flash, render_template, request

from network import Network

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

net = Network()
init_plot = net.generate_init_plot()

@app.route('/')
def render_init():
    return render_template('dashboard.html', include_plotlyjs=False,  plot=Markup(init_plot))

@app.route('/dashboard.html')
def index():
    return render_template('dashboard.html', include_plotlyjs=False,  plot=Markup(init_plot))

@app.route('/prediction.html')
def prediction_render():
    return render_template('prediction.html')

@app.route('/updation.html')
def updation_render():
    return render_template('updation.html')

@app.route('/profile.html')
def profile_render():
    return render_template('profile.html')

@app.route('/hello', methods=['POST'])
def hello():
    if request.form['submit'] == 'predict':
        pr = request.form['pr']
        predict = net.predict(str(pr))
        return render_template('prediction.html', predicted=predict)
        
    elif request.form['submit'] == 'update':
        px = request.form['px']
        py = request.form['py']
        x = str(px)

        new_plot = net.retrain_model(x, py)
        
        return render_template('updation.html', include_plotlyjs=False,  updated_plot=Markup(new_plot))

if __name__ == '__main__':
    app.run(debug=True) 
