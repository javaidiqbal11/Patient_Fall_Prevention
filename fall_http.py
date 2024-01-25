import time
import requests
import ipc


def fall_alert_sender(confidence):
    # global input_json
    # userid = uid[0]
    # req = ""
    # df = pd.read_csv("timer.csv")
    # start = int(df['timer'].tolist()[0])
    # time_diff = int(time.time()-start)

    # if time_diff < 25:
    #
    #     # requests.post("http://localhost:5555/fallalert", struct.pack("!i", confidence))
    #
    #     return "Within 25 Sec"

    # msg = "Please stay on the bed. If you get out of the bed, you can fall and get injured. Do you want me to request your nurse to visit you Please answer yes or no"
    types = ""
    if int(confidence) == 1 or int(confidence) == 2:
        types = "type_101"
        # input_json = {
        #     "MessageType": "Fall Alert",
        #     "IsBroadCastMessage": "False",
        #     "message": {
        #       #  "CustomerId": userid,
        #         "CustomerName": "XX",
        #         "MessageContent": {
        #             "question": msg,
        #             "types": types
        #         }
        #     }
        # }

    if int(confidence) == 3 or int(confidence) == 4:
        types = "type_102"
        # input_json = {
        #     "MessageType": "Fall Alert",
        #     "IsBroadCastMessage": "False",
        #     "message": {
        #        # "CustomerId": userid,
        #         "CustomerName": "XX",
        #         "MessageContent": {
        #             "question": msg,
        #             "types": types
        #         }
        #     }
        # }

        # print(input_json)
    try:
        start = time.time()
        # req = requests.post("http://localhost:5555/fallalert", json=input_json)
        # req = requests.post("http://localhost:5555/fallalert", data=str('FallAlertType:{}'.format(types)))
        # requests.post("http://localhost:5555/fallalert", struct.pack("!i", confidence))

        ipc.SendMessage(str('FallAlertType:{}'.format(types)), False)

        # print("HTTP time is:", {time.time()-start})
        # print("Res : ", req.json())
    except:
        pass

    # df1 = pd.DataFrame({"timer": str(time.time())}, index=[0])
    # df1.to_csv("timer.csv")

    # return req

# fall_alert_sender(1, 500)
