import os
import datetime as dt
import collections
import pickle
import cv2
import numpy as np
import random
import seaborn as sns
from matplotlib import pyplot as plt
import copy
from scipy.spatial import distance
from keras.optimizers import Adam
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss


def build_SSD(weight_path, class_list, img_dimension):
    '''
    --input--
    weight_path: path of the pre-trained weights,
    class_list: list of pre-trained classes,
    img_dimension: tupule of (img_height,img_width)
    --output--
    model: compiled SSD model
    classes: dict(class:index)
    classes_rev: dict(index:class)
    '''
    model = ssd_512(image_size=(img_dimension[0], img_dimension[1], 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0,
                                                                               2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 128, 256, 512],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)
    classes = {class_list[i]: i for i in range(0, len(class_list))}
    classes_rev = {i: class_list[i] for i in range(0, len(class_list))}

    model.load_weights(weight_path, by_name=True)

    adam = Adam(lr=0.001, epsilon=1e-08)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    return model, classes, classes_rev


def filter_by_classes(results, classes, labels=None):
    '''
    --input--
    results: prediction result from SSD_predict,
    classes: dict(class:index),
    labels: list of classes to filter the result by,
    --output--
    results: filtered result
    '''
    filtered_result = []
    for i in results:
        if i[0] in [classes[i] for i in labels]:
            filtered_result.append(i)
    results = filtered_result
    return results


def video_background(video_path, alpha):
    '''
    --input--
    video_path: path to the uploaded video,
    alpha: alpha value for moving average
    --output--
    background image with moving objects filtered out,
    vid_height: height of the input video,
    vid_width: width of the input video,
    FPS: frame per second of the input video,
    length: total duration of the input video (sec)
    '''
    cap = cv2.VideoCapture(video_path)

    frame_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    length = frame_numbers / FPS

    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    average_img = np.float32(img)

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.accumulateWeighted(img, average_img, alpha)
            background = cv2.convertScaleAbs(average_img)
        else:
            break
    cap.release()

    return background, vid_height, vid_width, FPS, length


def save_frame_range_video(video_path, saving_video_file_name, start_sec=0, end_sec=None):
    '''
    --input--
    video_path: path to the uploaded video,
    saving_video_file_name: saving file name of the video clip,
    start_sec: start time of the video clip (in sec),
    end_sec: end time of the video clip (in sec); if not defined -- end of the video
    --output--
    saved video clip (.avi)
    '''
    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    frame_numbers = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_numbers / FPS

    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if cap.isOpened() is False:
        print('cannot read a video')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join('static', saving_video_file_name + '.avi'),
                          fourcc, FPS, (vid_width, vid_height))

    if end_sec is None:
        end_sec = duration

    while (cap.isOpened() and frame_count < np.floor(end_sec * FPS)):
        _, frame = cap.read()
        if frame_count < np.floor(start_sec * FPS):
            frame_count += 1
            continue
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


color_palatte = {}
color_palatte_norm = {}


def color_by_index(idx):
    '''
    generate unique (r,g,b) color code per customer index [0,255]
    --input--
    idx: customer index,
    --output--
    color code
    '''
    if idx not in color_palatte.keys():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_palatte[idx] = (r, g, b)
    return color_palatte[idx]


def color_by_index_norm(idx):
    '''
    normalize the result of color_by_index to [0,1]
    --input--
    idx: customer index,
    --output--
    normalized color code
    '''
    r, g, b = color_palatte[idx]
    color_palatte_norm[idx] = (b / 255, g / 255, r / 255)
    return color_palatte_norm[idx]


def motion_tracking(video_path, initial_time, model, classes, video_name, file_name, skip_frame, min_dist_thresh, removing_thresh, confi_thresh, start_sec=0, end_sec=None):
    '''
    --input--
    video_path: path to the uploaded video,
    initial_time: beginning time of the video,
    model: SSD model,
    classes: dict(class:index),
    video_name: file name of the processed video,
    file_name: file name of the entire motion tracking history,
    skip_frame: number of frames to skip tracking,
    min_dist_thresh: minimum euclidian distance to differentiate two different objects,
    removing_thresh: number of the identicle position history to determine the object left the scene,
    confi_thresh: confidence threshold for SSD object detection [0,1],
    start_sec: start of the video to be processed (seconds),
    end_sec: end of the video to be processed (seconds)
    --output--
    sorted_archive: dictionary of the full motion history sorted by the created timestamp,
    background: background image with moving objects filtered out,
    length: total duration of the input video (sec)
    '''
    background, vid_height, vid_width, FPS, length = video_background(
        video_path, alpha=0.005)
    height_fix_factor = vid_height / 512
    width_fix_factor = vid_width / 512
    label_to_track = 'person'

    frame_count = 0

    print('processing starts at {:.2f} sec'.format(start_sec))

    cap = cv2.VideoCapture(video_path)

    if cap.isOpened() is False:
        print('cannot read a video')

    track_history = {}
    moving_tracker = {}
    archive = {}
    customer_idx = 1  # customer id

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join('static', video_name + '.avi'), fourcc,
                          FPS, (vid_width, vid_height))

    if end_sec is None:
        end_sec = length

    while cap.isOpened() and frame_count < np.floor(end_sec * FPS):

        ret, frame = cap.read()

        if frame_count < np.floor(start_sec * FPS):
            frame_count += 1
            continue

        if ret:

            frame = np.asarray(frame)
            orig_frame = np.copy(frame)

            if frame_count % skip_frame == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (512, 512))
                frame = np.expand_dims(frame, 0)
                current_centroids = []  # reset current centroid list at every processing frame

                # for the very first frame
                if not track_history:
                    result = model.predict(frame)
                    results = result[0][result[0, :, 1] >= confi_thresh]

                    results = filter_by_classes(
                        results, classes, labels=[label_to_track])

                    for r in results:
                        r[2] = xmin = int(r[2] * width_fix_factor)
                        r[3] = ymin = int(r[3] * height_fix_factor)
                        r[4] = xmax = int(r[4] * width_fix_factor)
                        r[5] = ymax = int(r[5] * height_fix_factor)
                        centroid = (int(np.mean((xmin, xmax))),
                                    int(np.mean((ymin, ymax))))

                        # add the list of positions
                        track_history[customer_idx] = [
                            initial_time + dt.timedelta(seconds=frame_count / FPS), [centroid]]
                        # initialize moving racker
                        moving_tracker[customer_idx] = 0
                        customer_idx += 1

                else:
                    result = model.predict(frame)
                    results = result[0][result[0, :, 1] >= confi_thresh]
                    results = filter_by_classes(
                        results, classes, labels=[label_to_track])

                    for r in results:
                        r[2] = xmin = int(r[2] * width_fix_factor)
                        r[3] = ymin = int(r[3] * height_fix_factor)
                        r[4] = xmax = int(r[4] * width_fix_factor)
                        r[5] = ymax = int(r[5] * height_fix_factor)
                        centroid = (int(np.mean((xmin, xmax))),
                                    int(np.mean((ymin, ymax))))
                        current_centroids.append(centroid)

                    track_history_temp = copy.deepcopy(track_history)
                    track_history_key_temp = copy.deepcopy(
                        list(track_history.keys()))

                    # comparison
                    for cent in current_centroids:
                        min_dist = min_dist_thresh
                        min_label = None
                        for label in track_history_key_temp:
                            dist = distance.euclidean(
                                cent, track_history_temp[label][-1][-1])
                            if dist < min_dist:
                                min_dist = dist
                                min_label = label

                        # for same label centroid
                        if min_label is not None:
                            if min_dist == 0:  # if object not moved, increase the moving tracker counter by 1
                                moving_tracker[min_label] += 1
                            else:  # if moved, reset the tracker counter
                                moving_tracker[min_label] = 0
                            track_history_temp[min_label][-1].append(cent)
                            track_history_key_temp.remove(min_label)

                        # min_label is NONE --> NEW object in the scene
                        else:
                            track_history_temp[customer_idx] = [
                                initial_time + dt.timedelta(seconds=frame_count / FPS), [cent]]
                            moving_tracker[customer_idx] = 0
                            customer_idx += 1
                    # object hidden or exit
                    if track_history_key_temp:
                        for left_over in track_history_key_temp:
                            moving_tracker[left_over] += 1
                            track_history_temp[left_over][-1].append(
                                track_history_temp[left_over][-1][-1])

                    track_history = track_history_temp  # update the history

                # generate orig_frame based on track_history
                for idx, loc in track_history.items():
                    cv2.circle(orig_frame, loc[-1][-1],
                               10, color_by_index(idx), cv2.FILLED)

                # move the unmoving objects to the archive dictionary
                moving_tracker_temp = copy.deepcopy(moving_tracker)
                for obj, counter in moving_tracker_temp.items():
                    if counter == removing_thresh:
                        archive[obj] = [track_history[obj][0],
                                        track_history[obj][-1][:-removing_thresh]]
                        del track_history[obj]
                        del moving_tracker[obj]
                print('>', end='')

            # in-between frames
            else:
                for idx, loc in track_history.items():
                    cv2.circle(orig_frame, loc[-1][-1],
                               10, color_by_index(idx), cv2.FILLED)

            out.write(orig_frame)
            frame_count += 1

        else:
            break
    print('\n')
    print('proccesing finished at {:.2f} sec'.format(frame_count / FPS))
    print('Total time processed: {:.2f} sec'.format(
        frame_count / FPS - start_sec))
    cap.release()
    out.release()

    # after all, move all to archive
    moving_tracker_temp = copy.deepcopy(moving_tracker)
    for obj, counter in moving_tracker_temp.items():
        archive[obj] = [track_history[obj][0], track_history[obj][-1]]
        del track_history[obj]
        del moving_tracker[obj]

    sorted_archive = sorted(
        archive.items(), key=lambda kv: kv[1][0], reverse=False)
    sorted_archive = collections.OrderedDict(sorted_archive)

    pickle.dump(sorted_archive, open(file_name + '.pkl', 'wb'))
    pickle.dump(background, open('background' + '.pkl', 'wb'))

    print('processed video saved as: "{}.avi"'.format(video_name))
    print('file saved as: "{}.txt"'.format(file_name))

    return sorted_archive, background, length


def time_slice(full_dict, dt_obj=None, flag=None):
    '''
    --input--
    full_dict: dictionary of the full motion history,
    dt_obj: list of datetime object (1 or 2 elements),
    flag: comparison flags (to be chosen among 'before','after','in','out')
    ** when len(dt_obj) is 1, flags should be either 'before' or 'after' ||
    ** when len(dt_obj) is 2, flags should be either 'in' or 'out' ||
    --output--
    sliced_archive: sliced dictionary of the motion history
    '''
    sliced_archive = {}

    if dt_obj[-1] == '':
        if flag == 'after':
            for k, v in full_dict.items():
                if v[0] > dt_obj[0]:
                    sliced_archive[k] = full_dict[k]
            return sliced_archive

        if flag == 'before':
            for k, v in full_dict.items():
                if v[0] < dt_obj[0]:
                    sliced_archive[k] = full_dict[k]
            return sliced_archive
        else:
            sliced_archive = 'wrong method!'
            return sliced_archive

    else:
        if flag == 'in':
            for k, v in full_dict.items():
                if v[0] > dt_obj[0] and v[0] < dt_obj[1]:
                    sliced_archive[k] = full_dict[k]

            return sliced_archive

        if flag == 'out':
            for k, v in full_dict.items():
                if v[0] < dt_obj[0] or v[0] > dt_obj[1]:
                    sliced_archive[k] = full_dict[k]

            return sliced_archive
        else:
            sliced_archive = 'wrong method!'
            return sliced_archive


def contour_draw(dict_input, background, img_name, alpha=0.6, n_levels=5, figsize=(20, 10)):
    '''
    --input--
    dict_input: dictionary of the motion history,
    background: background image with moving objects filtered out,
    img_name = saving file name of the contour image,
    alpha: alpha value of the background image [0,1],
    n_levels: number of contour lines,
    figsize: figure size
    --output--
    conture image
    '''
    xx = []
    yy = []
    for key in dict_input.keys():
        for centroid in dict_input[key][1]:
            xx.append(centroid[0])
            yy.append(centroid[1])
    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(background, alpha=alpha)
    sns.kdeplot(xx, yy, ax=ax, n_levels=n_levels)
    plt.axis('off')
    fig.savefig(os.path.join('static', img_name + '.png'))


def tracjactory_draw(dict_input, background, img_name, alpha=0.3, markersize=20, lw=10, figsize=(15, 8)):
    '''
    --input--
    dict_input: dictionary of the motion history,
    background: background image with moving objects filtered out,
    img_name = saving file name of the trajectory image
    alpha: alpha value of the marker [0,1],
    markersize: size of the markers,
    lw: line width between the data markers,
    figsize: figure size
    --output--
    motion trajactory image
    '''
    plt.figure(figsize=figsize)
    ax = plt.gca()
    plt.imshow(background, alpha=0.5)
    for key in dict_input.keys():
        x = []
        y = []
        for centroid in dict_input[key][1]:
            x.append(centroid[0])
            y.append(centroid[1])
        ax.plot(x, y, marker='o', c=color_by_index_norm(key),
                alpha=alpha, markersize=markersize, lw=lw)
    plt.axis('off')
    plt.savefig(os.path.join('static', img_name + '.png'))
