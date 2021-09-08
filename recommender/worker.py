from celery import Celery
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from user_item_data import userItemData
import redis

r = redis.Redis(host='redis')
r.set('test', 'ololo')

logger = get_task_logger(__name__)
logger.error('Worker started')
logger.info('Worker started')
print('sdhbvsvbskjvb;')
ui_data = userItemData('./data/raw_ratings.csv')
celery_app = Celery('worker', backend='redis://redis:6379/0', broker='redis://redis:6379/0')  # broker and database is redis

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
