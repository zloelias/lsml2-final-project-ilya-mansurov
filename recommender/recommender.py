from celery import Celery
from celery.backends.redis import RedisBackend
from flask import Flask, request, jsonify, abort
import json
import logging
import  numpy as np
import requests
from user_item_data import userItemData
#from worker import celery_app

celery_app = Celery('recommender', backend='redis://redis:6379/0', broker='redis://redis:6379/0')  # broker and database is redis
app = Flask(__name__) # Main object of the Flask application

topK = 10
logging.basicConfig(level=logging.DEBUG)
ui_data = userItemData('./data/raw_ratings.csv')

@celery_app.task()
def predict(X):
    r = requests.post(
        'http://model:8080/invocations',
        headers={"Content-Type": "application/json"},
        data=json.dumps(X)
    )
    if r.status_code == 200:
        result = ui_data.get_items_id_by_items_ind(
            np.array(X['data'])[np.array(r.json()).reshape(-1).argsort()[::-1][:topK], 1]
        ).tolist()
    else:
        result = None
    return result


@app.route ('/') # Function handler for /
def hello():
    return "Hello, from Flask" # Return the string as a response

@app.route('/count_recs_by_user/<user_id>')
def model_handler(user_id):
    user_ind = ui_data.get_user_ind_by_user_id(user_id)
    if not user_id:
        # TODO return some base not personal recommendation here, top rated, most viewed etc
        abort(404)

    else:
        X = ui_data.get_not_rated_items_ind_by_user_ind(user_ind)
        #print(X)
        task = predict.delay({'data': X.tolist()})
        logging.debug(f'set task: {task.id}, {task.ready()}')

        response = {
            "task_id": task.id
        }

        return json.dumps(response)  # encode the response to JSON

@app.route('/get_recs/<task_id>')
def model_check_handler(task_id):
    task = celery_app.AsyncResult(task_id, app=celery_app, backend=RedisBackend(app=celery_app, url='redis://redis:6379/0'))
    if task.ready():
        # TODO check if result is None return some base not personal recommendation here, top rated, most viewed etc
        response = {
            "status": "DONE",
            "result": task.result
        }
    else:
        response = {
            "status": "IN_PROGRESS"
        }
    return json.dumps(response)



if __name__ == "__main__":
    app.run(host="0.0.0.0")
