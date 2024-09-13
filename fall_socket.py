# Server and Client Socket Program 
import time
import ipc
import logging
import datetime


def fall_alert_sending(confidence):
    types = ""
    if int(confidence) == 1 or int(confidence) == 2:
        types = "type_101"

    if int(confidence) == 3 or int(confidence) == 4:
        types = "type_102"

    try:
        start = time.time()
        ipc.SendMessage(str('FallAlertType:{}'.format(types)), False)
        logging.info("{} Connected Socket".format(datetime.datetime.now()))

    except:
        pass


# fall_alert_sending(1, 500)
