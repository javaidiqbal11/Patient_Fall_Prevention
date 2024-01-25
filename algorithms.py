import logging
import time
from matplotlib import pyplot as plt
from vis.visual import write_on_image, activity_dict, visualise_tracking
from vis.processor import Processor
from vis.inv_pendulum import *
import re
import pandas as pd
from model.model import LSTMModel
import torch
import math
import datetime
from fall_http import fall_alert_sender
from fall_socket import fall_alert_sending


def get_source(args):
    tagged_df = None

    if args.video is None:
        cam = cv2.VideoCapture(0)

    else:
        logging.info("{} Video source: {}".format(datetime.datetime.now(), args.video))
        cam = cv2.VideoCapture(args.video)

        if isinstance(args.video, str):
            vid = [int(s) for s in re.findall(r'\d+', args.video)]
            if len(vid) == 5:
                tagged_df = pd.read_csv("dataset/CompleteDataSet.csv", usecols=[
                    "TimeStamps", "Subject", "Activity", "Trial", "Tag"], skipinitialspace=True)
                tagged_df = tagged_df.query(
                    f'Subject == {vid[1]} & Activity == {vid[0]} & Trial == {vid[2]}')
    img = cam.read()[1]

    logging.info('Image shape:{}'.format(str(img.shape)))

    return cam, tagged_df


def resize(img, resize, resolution):
    if resize is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in resize.split('x')]
    width_height = (int(width * resolution // 16) * 16,
                    int(height * resolution // 16) * 16)
    return width, height, width_height


def extract_keypoints_parallel(queue, args, self_counter, other_counter, consecutive_frames, event):
    print(queue, args, "%%", self_counter, other_counter, "**", consecutive_frames, event)
    try:
        logging.info("{} Extractor started".format(datetime.datetime.now()))
        cam, tagged_df = get_source(args)
        ret_val, img = cam.read()
    except Exception as e:
        queue.put(None)
        event.set()
        print('Exception occurred:', e)
        return

    width, height, width_height = resize(img, args.resize, args.resolution)
    logging.info(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    output_video = None

    frame = 0
    fps = 0
    t0 = time.time()
    count = 0
    while not event.is_set():
        count += 1
        logging.info("{} >>event loop started<< count={}".format(datetime.datetime.now(), count))
        if args.num_cams == 2 and (self_counter.value > other_counter.value):
            continue

        ret_val, img = cam.read()
        frame += 1
        self_counter.value += 1
        if tagged_df is None:
            curr_time = time.time()
        else:
            curr_time = tagged_df.iloc[frame - 1]['TimeStamps'][11:]
            curr_time = sum(x * float(t) for x, t in zip([3600, 60, 1], curr_time.split(":")))

        if img is None:
            print('no more images captured')
            print(args.video, curr_time, sep=" ")
            if not event.is_set():
                event.set(0)
            break

        logging.info("{} setting image".format(datetime.datetime.now()))
        img = cv2.resize(img, (width, height))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        logging.info("{} processor started".format(datetime.datetime.now()))
        keypoint_sets, bb_list, width_height = processor_singleton.single_image(img)

        logging.info("{} creating delay by fall alert".format(datetime.datetime.now()))
        time.sleep(0.70)  # Drop FPS for the video
        logging.info("{} processer ended".format(datetime.datetime.now()))
        assert bb_list is None or (type(bb_list) == list)
        if bb_list:
            assert type(bb_list[0]) == tuple
            assert type(bb_list[0][0]) == tuple
        # assume bb_list is of the form [(x1,y1),(x2,y2)),etc.]
        logging.info("{} setting image complete".format(datetime.datetime.now()))
        if args.coco_points:
            keypoint_sets = [keypoints.tolist() for keypoints in keypoint_sets]
        else:
            anns = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
            ubboxes = [(np.asarray([width, height]) * np.asarray(ann[1])).astype('int32')
                       for ann in anns]
            lbboxes = [(np.asarray([width, height]) * np.asarray(ann[2])).astype('int32')
                       for ann in anns]
            bbox_list = [(np.asarray([width, height]) * np.asarray(box)).astype('int32') for box in bb_list]
            uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
            lhist_list = [get_hist(img, bbox) for bbox in lbboxes]
            keypoint_sets = [{"keypoints": keyp[0], "up_hist": uh, "lo_hist": lh, "time": curr_time, "box": box}
                             for keyp, uh, lh, box in zip(anns, uhist_list, lhist_list, bbox_list)]

            cv2.polylines(img, ubboxes, True, (255, 0, 0), 2)
            cv2.polylines(img, lbboxes, True, (0, 255, 0), 2)
            for box in bbox_list:
                cv2.rectangle(img, tuple(box[0]), tuple(box[1]), ((0, 0, 255)), 2)

            fpsLimit = 1

        dict_vis = {"img": img, "keypoint_sets": keypoint_sets, "width": width, "height": height,
                    "vis_keypoints": args.joints,
                    "vis_skeleton": args.skeleton, "CocoPointsOn": args.coco_points,
                    "tagged_df": {"text": f"", "color": [0, 0, 0]}}
        # "tagged_df": {"text": f"Avg FPS: {frame // (time.time() - t0)}, Frame: {frame}", "color": [0, 0, 0]}}

        queue.put(dict_vis)
        logging.info("{} <event loop ended> queue size:{}".format(datetime.datetime.now(), queue.qsize()))

    queue.put(None)
    return


def show_tracked_img(img_dict, ip_set, num_matched, output_video, args):
    img = img_dict["img"]
    tagged_df = img_dict["tagged_df"]
    keypoints_frame = [person[-1] for person in ip_set]
    img = visualise_tracking(img=img, keypoint_sets=keypoints_frame, width=img_dict["width"], height=img_dict["height"],
                             num_matched=num_matched, vis_keypoints=img_dict["vis_keypoints"],
                             vis_skeleton=img_dict["vis_skeleton"],
                             CocoPointsOn=False)

    img = write_on_image(img=img, text=tagged_df["text"],
                         color=tagged_df["color"])

    if output_video is None:
        if args.save_output:
            vidname = args.video.split('/')
            output_video = cv2.VideoWriter(filename='/'.join(vidname[:-1]) + '/out' + vidname[-1][:-3] + 'avi',
                                           fourcc=cv2.VideoWriter_fourcc(*'MP42'),
                                           fps=args.fps, frameSize=img.shape[:2][::-1])
            logging.info(
                "{} Saving the output video at {} with {} frames per seconds".format(datetime.datetime.now(),
                                                                                     args.out_path, args.fps))
        else:
            output_video = None
            logging.info(f'Not saving the output video')
    else:
        output_video.write(img)
    return img, output_video


def remove_wrongly_matched(matched_1, matched_2):
    unmatched_idxs = []
    i = 0

    for ip1, ip2 in zip(matched_1, matched_2):
        # each of these is a set of the last t frames of each matched person
        correlation = cv2.compareHist(last_valid_hist(ip1)["up_hist"], last_valid_hist(ip2)["up_hist"],
                                      cv2.HISTCMP_CORREL)
        if correlation < 0.5 * HIST_THRESH:
            unmatched_idxs.append(i)
        i += 1

    return unmatched_idxs


def match_unmatched(unmatched_1, unmatched_2, lstm_set1, lstm_set2, num_matched):
    new_matched_1 = []
    new_matched_2 = []
    new_lstm1 = []
    new_lstm2 = []
    final_pairs = [[], []]

    if not unmatched_1 or not unmatched_2:
        return final_pairs, new_matched_1, new_matched_2, new_lstm1, new_lstm2

    new_matched = 0
    correlation_matrix = - np.ones((len(unmatched_1), len(unmatched_2)))
    dist_matrix = np.zeros((len(unmatched_1), len(unmatched_2)))
    for i in range(len(unmatched_1)):
        for j in range(len(unmatched_2)):
            correlation_matrix[i][j] = cv2.compareHist(last_valid_hist(unmatched_1[i])["up_hist"],
                                                       last_valid_hist(unmatched_2[j])["up_hist"], cv2.HISTCMP_CORREL)
            dist_matrix[i][j] = np.sum(
                np.absolute(last_valid_hist(unmatched_1[i])["up_hist"] - last_valid_hist(unmatched_2[j])["up_hist"]))

    freelist_1 = [i for i in range(len(unmatched_1))]
    pair_21 = [-1] * len(unmatched_2)
    unmatched_1_preferences = np.argsort(-correlation_matrix, axis=1)
    unmatched_indexes1 = [0] * len(unmatched_1)
    finish_array = [False] * len(unmatched_1)
    while freelist_1:
        um1_idx = freelist_1[-1]
        if finish_array[um1_idx] == True:
            freelist_1.pop()
            continue
        next_unasked_2 = unmatched_1_preferences[um1_idx][unmatched_indexes1[um1_idx]]
        if pair_21[next_unasked_2] == -1:
            pair_21[next_unasked_2] = um1_idx
            freelist_1.pop()
        else:
            curr_paired_2 = pair_21[next_unasked_2]
            if correlation_matrix[curr_paired_2][next_unasked_2] < correlation_matrix[um1_idx][next_unasked_2]:
                pair_21[next_unasked_2] = um1_idx
                freelist_1.pop()
                if not finish_array[curr_paired_2]:
                    freelist_1.append(curr_paired_2)

        unmatched_indexes1[um1_idx] += 1
        if unmatched_indexes1[um1_idx] == len(unmatched_2):
            finish_array[um1_idx] = True

    for j, i in enumerate(pair_21):
        if correlation_matrix[i][j] > HIST_THRESH:
            final_pairs[0].append(i + num_matched)
            final_pairs[1].append(j + num_matched)
            new_matched_1.append(unmatched_1[i])
            new_matched_2.append(unmatched_2[j])
            new_lstm1.append(lstm_set1[i])
            new_lstm2.append(lstm_set2[j])

    return final_pairs, new_matched_1, new_matched_2, new_lstm1, new_lstm2


def alg2_sequential(queues, argss, consecutive_frames, event):
    t1, t2 = 0, 0
    last_req_time = 0
    repeated_last_req_time = 0  # time.time()
    minimum_request_delay = 0  # seconds
    repeated_request_delay = 0  # seconds

    logging.info("{} alg2_seq started".format(datetime.datetime.now()))
    model = LSTMModel(h_RNN=48, h_RNN_layers=2, drop_p=0.1, num_classes=7)
    model.load_state_dict(torch.load('model/lstm_weights.sav', map_location=argss[0].device))
    model.eval()
    output_videos = [None for _ in range(argss[0].num_cams)]
    t0 = time.time()
    feature_plotters = [[[], [], [], [], []] for _ in range(argss[0].num_cams)]
    ip_sets = [[] for _ in range(argss[0].num_cams)]
    lstm_sets = [[] for _ in range(argss[0].num_cams)]
    max_length_mat = 300
    num_matched = 0
    prev = 0
    if not argss[0].plot_graph:
        max_length_mat = consecutive_frames
    else:
        f, ax = plt.subplots()
        move_figure(f, 800, 100)
    window_names = [args.video if isinstance(args.video, str) else 'Cam ' + str(args.video) for args in argss]
    [cv2.namedWindow(window_name) for window_name in window_names]
    head_time = time.time()
    logging.info("{} head time:{}, {}".format(head_time, t0, repeated_last_req_time))
    while True:
        if not any(q.empty() for q in queues):
            logging.info("{} not any(q.empty".format(datetime.datetime.now()))
            dict_frames = [q.get() for q in queues]

            if any([(dict_frame is None) for dict_frame in dict_frames]):
                if not event.is_set():
                    event.set()
                break

            if cv2.waitKey(1) == 27 or any(
                    cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 for window_name in window_names):
                if not event.is_set():
                    event.set()

            kp_frames = [dict_frame["keypoint_sets"] for dict_frame in dict_frames]
            if argss[0].num_cams == 1:
                logging.info("{} num cam:1".format(datetime.datetime.now()))
                num_matched, new_num, indxs_unmatched = match_ip(ip_sets[0], kp_frames[0], lstm_sets[0], num_matched,
                                                                 max_length_mat)
                valid1_idxs, prediction = get_all_features(ip_sets[0], lstm_sets[0], model)

                logging.info("{} calling fall alert".format(datetime.datetime.now()))

                time_passed = time.time() - last_req_time
                confidence_type = ""
                if True or time_passed > minimum_request_delay:
                    last_req_time = time.time()
                    if prediction + 5 == 8:
                        confidence = 4  # Emergency
                        confidence_type = "type_102"
                    elif prediction + 5 == 5:
                        confidence = 3  # Highest Alert
                        confidence_type = "type_102"
                    elif prediction + 5 == 12:
                        confidence = 2  # Moderate
                        confidence_type = "type_101"
                    elif prediction + 5 == 7:
                        confidence = 1  # Low Risk
                        confidence_type = "type_101"
                    else:
                        logging.info("{} did not called fall alert".format(datetime.datetime.now()))
                        confidence = 0
                        pass

                    logging.info("{} confidence = {}".format(datetime.datetime.now(), confidence))
                    repeated_time_passed = time.time() - repeated_last_req_time
                    if confidence and repeated_time_passed > repeated_request_delay:
                        if confidence == prev:
                            pass
                        else:
                            prev = confidence
                        repeated_last_req_time = time.time()
                        # fall_alert(prev, uid)
                        # fall_alert_sender(prev)
                        fall_alert_sending(prev)

                        # logging.info("{} HTTP Sending".format(datetime.datetime.now()))
                        # try:
                        #     requests.post("http://localhost:5555/fallalert", data=str('FallAlertType:{}'.format(confidence_type)))
                        # except Exception as e:
                        #     logging.info("{} HTTP Failed: {}".format(datetime.datetime.now(), e))


                    else:
                        logging.info("{} skipped repeated request".format(datetime.datetime.now()))
                else:
                    logging.info("{} skipped fall alert - last alert < {} seconds ago".format(datetime.datetime.now(),
                                                                                              time_passed))

                dict_frames[0]["tagged_df"]["text"] += f" Confidence: {confidence} " if confidence else ""

                img, output_videos[0] = show_tracked_img(dict_frames[0], ip_sets[0], num_matched, output_videos[0],
                                                         argss[0])
                cv2.imshow(window_names[0], img)
                logging.info("{} imshow".format(datetime.datetime.now()))

            elif argss[0].num_cams == 2:
                logging.info("{} 2 num_cams".format(datetime.datetime.now()))
                num_matched, new_num, indxs_unmatched1 = match_ip(ip_sets[0], kp_frames[0], lstm_sets[0], num_matched,
                                                                  max_length_mat)
                assert (new_num == len(ip_sets[0]))
                for i in sorted(indxs_unmatched1, reverse=True):
                    elem = ip_sets[1][i]
                    ip_sets[1].pop(i)
                    ip_sets[1].append(elem)
                    elem_lstm = lstm_sets[1][i]
                    lstm_sets[1].pop(i)
                    lstm_sets[1].append(elem_lstm)
                num_matched, new_num, indxs_unmatched2 = match_ip(ip_sets[1], kp_frames[1], lstm_sets[1], num_matched,
                                                                  max_length_mat)

                for i in sorted(indxs_unmatched2, reverse=True):
                    elem = ip_sets[0][i]
                    ip_sets[0].pop(i)
                    ip_sets[0].append(elem)
                    elem_lstm = lstm_sets[0][i]
                    lstm_sets[0].pop(i)
                    lstm_sets[0].append(elem_lstm)

                matched_1 = ip_sets[0][:num_matched]
                matched_2 = ip_sets[1][:num_matched]

                unmatch_previous = remove_wrongly_matched(matched_1, matched_2)
                if unmatch_previous:
                    print(unmatch_previous)

                for i in sorted(unmatch_previous, reverse=True):
                    elem1 = ip_sets[0][i]
                    elem2 = ip_sets[1][i]
                    ip_sets[0].pop(i)
                    ip_sets[1].pop(i)
                    ip_sets[0].append(elem1)
                    ip_sets[1].append(elem2)
                    elem_lstm1 = lstm_sets[0][i]
                    lstm_sets[0].pop(i)
                    lstm_sets[0].append(elem_lstm1)
                    elem_lstm2 = lstm_sets[1][i]
                    lstm_sets[1].pop(i)
                    lstm_sets[1].append(elem_lstm2)
                    num_matched -= 1

                unmatched_1 = ip_sets[0][num_matched:]
                unmatched_2 = ip_sets[1][num_matched:]

                new_pairs, new_matched1, new_matched2, new_lstm1, new_lstm2 = match_unmatched(
                    unmatched_1, unmatched_2, lstm_sets[0], lstm_sets[1], num_matched)

                new_p1 = new_pairs[0]
                new_p2 = new_pairs[1]

                for i in sorted(new_p1, reverse=True):
                    ip_sets[0].pop(i)
                    lstm_sets[0].pop(i)
                for i in sorted(new_p2, reverse=True):
                    ip_sets[1].pop(i)
                    lstm_sets[1].pop(i)

                ip_sets[0] = ip_sets[0][:num_matched] + new_matched1 + ip_sets[0][num_matched:]
                ip_sets[1] = ip_sets[1][:num_matched] + new_matched2 + ip_sets[1][num_matched:]
                lstm_sets[0] = lstm_sets[0][:num_matched] + new_lstm1 + lstm_sets[0][num_matched:]
                lstm_sets[1] = lstm_sets[1][:num_matched] + new_lstm2 + lstm_sets[1][num_matched:]

                num_matched = num_matched + len(new_matched1)

                valid1_idxs, prediction1 = get_all_features(ip_sets[0], lstm_sets[0], model)
                valid2_idxs, prediction2 = get_all_features(ip_sets[1], lstm_sets[1], model)
                dict_frames[0]["tagged_df"]["text"] += f" Pred: {activity_dict[prediction1 + 5]}"

                dict_frames[1]["tagged_df"]["text"] += f" Pred: {activity_dict[prediction2 + 5]}"

                img1, output_videos[0] = show_tracked_img(dict_frames[0], ip_sets[0], num_matched, output_videos[0],
                                                          argss[0])
                img2, output_videos[1] = show_tracked_img(dict_frames[1], ip_sets[1], num_matched, output_videos[1],
                                                          argss[1])
                cv2.imshow(window_names[0], img1)
                cv2.imshow(window_names[1], img2)

                assert (len(lstm_sets[0]) == len(ip_sets[0]))
                assert (len(lstm_sets[1]) == len(ip_sets[1]))

            logging.info("{} not any(q.empty ended".format(datetime.datetime.now()))
            DEBUG = False

    cv2.destroyAllWindows()

    del model
    logging.info("{} end time".format(datetime.datetime.now()))
    return


def get_all_features(ip_set, lstm_set, model):
    logging.info("{} getting all features".format(datetime.datetime.now()))
    valid_idxs = []
    invalid_idxs = []
    predictions = [15] * len(ip_set)

    for i, ips in enumerate(ip_set):
        last1 = None
        last2 = None
        for j in range(-2, -1 * DEFAULT_CONSEC_FRAMES - 1, -1):
            if ips[j] is not None:
                if last1 is None:
                    last1 = j
                elif last2 is None:
                    last2 = j
        if ips[-1] is None:
            invalid_idxs.append(i)
        else:
            ips[-1]["features"] = {}
            ips[-1]["features"]["height_bbox"] = get_height_bbox(ips[-1])
            ips[-1]["features"]["ratio_bbox"] = FEATURE_SCALAR["ratio_bbox"] * get_ratio_bbox(ips[-1])

            body_vector = ips[-1]["keypoints"]["N"] - ips[-1]["keypoints"]["B"]
            ips[-1]["features"]["angle_vertical"] = FEATURE_SCALAR["angle_vertical"] * get_angle_vertical(body_vector)
            ips[-1]["features"]["log_angle"] = FEATURE_SCALAR["log_angle"] * np.log(
                1 + np.abs(ips[-1]["features"]["angle_vertical"]))

            if last1 is None:
                invalid_idxs.append(i)
            else:
                ips[-1]["features"]["re"] = FEATURE_SCALAR["re"] * get_rot_energy(ips[last1], ips[-1])
                ips[-1]["features"]["ratio_derivative"] = FEATURE_SCALAR["ratio_derivative"] * get_ratio_derivative(
                    ips[last1], ips[-1])
                if last2 is None:
                    invalid_idxs.append(i)
                else:
                    ips[-1]["features"]["gf"] = get_gf(ips[last2], ips[last1], ips[-1])
                    valid_idxs.append(i)

        xdata = []
        if ips[-1] is None:
            if last1 is None:
                xdata = [0] * len(FEATURE_LIST)
            else:
                for feat in FEATURE_LIST[:FRAME_FEATURES]:
                    xdata.append(ips[last1]["features"][feat])
                xdata += [0] * (len(FEATURE_LIST) - FRAME_FEATURES)
        else:
            for feat in FEATURE_LIST:
                if feat in ips[-1]["features"]:
                    xdata.append(ips[-1]["features"][feat])
                else:
                    xdata.append(0)

        xdata = torch.Tensor(xdata).view(-1, 1, 5)
        outputs, lstm_set[i][0] = model(xdata, lstm_set[i][0])
        if i == 0:
            prediction = torch.max(outputs.data, 1)[1][0].item()
            confidence = torch.max(outputs.data, 1)[0][0].item()

            fpd = True
            if fpd:
                if prediction in [1, 2, 3, 5]:
                    lstm_set[i][3] -= 1
                    lstm_set[i][3] = max(lstm_set[i][3], 0)

                    if lstm_set[i][2] < EMA_FRAMES:
                        if ips[-1] is not None:
                            lstm_set[i][2] += 1
                            lstm_set[i][1] = (lstm_set[i][1] * (lstm_set[i][2] - 1) + get_height_bbox(ips[-1])) / \
                                             lstm_set[i][2]
                    else:
                        if ips[-1] is not None:
                            lstm_set[i][1] = (1 - EMA_BETA) * get_height_bbox(ips[-1]) + EMA_BETA * lstm_set[i][1]

                elif prediction == 0:
                    if (ips[-1] is not None and lstm_set[i][1] != 0 and \
                        abs(ips[-1]["features"]["angle_vertical"]) < math.pi / 4) or confidence < 0.4:
                        prediction = 7
                    else:
                        lstm_set[i][3] += 1
                        if lstm_set[i][3] < DEFAULT_CONSEC_FRAMES // 4:
                            prediction = 7
                else:
                    lstm_set[i][3] -= 1
                    lstm_set[i][3] = max(lstm_set[i][3], 0)
            predictions[i] = prediction

    logging.info("{} completed getting features".format(datetime.datetime.now()))
    return valid_idxs, predictions[0] if len(predictions) > 0 else 15


def get_frame_features(ip_set, new_frame, re_matrix, gf_matrix, num_matched, max_length_mat=DEFAULT_CONSEC_FRAMES):
    match_ip(ip_set, new_frame, re_matrix, gf_matrix, max_length_mat)
    # return
    for i in range(len(ip_set)):
        if ip_set[i][-1] is not None:
            if ip_set[i][-2] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                    ip_set[i][-2], ip_set[i][-1]), max_length_mat)
            elif ip_set[i][-3] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                    ip_set[i][-3], ip_set[i][-1]), max_length_mat)
            elif ip_set[i][-4] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                    ip_set[i][-4], ip_set[i][-1]), max_length_mat)
            else:
                pop_and_add(re_matrix[i], 0, max_length_mat)
        else:
            pop_and_add(re_matrix[i], 0, max_length_mat)

    for i in range(len(ip_set)):
        if ip_set[i][-1] is not None:
            last1 = None
            last2 = None
            for j in [-2, -3, -4, -5]:
                if ip_set[i][j] is not None:
                    if last1 is None:
                        last1 = j
                    elif last2 is None:
                        last2 = j

            if last2 is None:
                pop_and_add(gf_matrix[i], 0, max_length_mat)
                continue

            pop_and_add(gf_matrix[i], get_gf(ip_set[i][last2], ip_set[i][last1],
                                             ip_set[i][-1]), max_length_mat)

        else:

            pop_and_add(gf_matrix[i], 0, max_length_mat)

    return
