import logging
import time
import requests
import threading
import datetime

testing_api = False


def request_task(url, json, headers):
    if testing_api:
        requests.post(url, json=json, headers=headers, cookies={"ngrok-skip-browser-warning": "any value"})
    else:
        logging.info("{} sending request 1.1".format(time.time()))
        requests.post(url, json=json, headers=headers)

        logging.info("{} sending request 1.2".format(time.time()))


def fire_and_forget(url, json, headers):
    logging.info("{} sending request 1".format(time.time()))
    threading.Thread(target=request_task, args=(url, json, headers)).start()
    logging.info("{} sending request 2".format(time.time()))


def fall_alert(confidence, uid):
    params = {
        "userid": uid[0],
        "confidence": confidence
    }

    logging.info("{} sending request start".format(time.time()))

    try:
        if testing_api:
            fire_and_forget("https://dc10-103-97-211-74.in.ngrok.io/test", json={"userid": uid[0],
                                                                                 "confidence": confidence,
                                                                                 "time": time.time()}, headers=None)
        else:
            logging.info("{} sending request to api".format(datetime.datetime()))

        logging.info("{} sent the request".format(time.time()))  # , str(req.status_code)))
    except Exception as e:

        logging.info("{} Request ended on {} has following errors {}".format(time.time(), e))
    # kafka API
    requests.post("https://ddxrx.ai/patient_fall_alert", json=params)

    # HTTP Server Configure
    # requests.post("http://localhost:5555/message", data=bytes(message, 'utf-8'))

    requests.post("")

    # Web browser API
    requests.post("https://jackson.ddxrx.com/api/confidence-insert.php", json=params)

    logging.info("{} sending request ended".format(time.time()))

# fall_alert(1)
