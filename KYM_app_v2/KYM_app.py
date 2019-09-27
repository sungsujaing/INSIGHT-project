from flask import Flask, render_template, request, session
import matplotlib
matplotlib.use('Agg')
import helper_func_app as hf
from keras import backend as K
import datetime as dt
import pickle
import os
import cv2

saving_video_file_name = 'original_video_clip'
video_name = 'processed_video_clip'
file_name = 'processed_file'

# Create the application object
app = Flask(__name__)


# we are now using these methods to get user input
@app.route('/', methods=["GET", "POST"])
def home_page():
  return render_template('index.html')


@app.route('/upload', methods=["GET", "POST"])
def KYM_upload():
  input_video = request.files['input_video']
  input_video.save('static/uploaded.mp4')
  return render_template("index.html",
                         my_form_result="uploaded")


@app.route('/results', methods=["GET", "POST"])
def KYM_results():

  input_start = request.args.get('input_start')
  input_end = request.args.get('input_end')

  if input_start == '':
    start = 0
  else:
    start = int(input_start)
  if input_end == '':
    end = None
  else:
    end = int(input_end)

  img_height = 512
  img_width = 512
  weights_path = 'VGG_VOC0712_SSD_512x512_iter_120000.h5'
  classes_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']

  model, classes, classes_rev = hf.build_SSD(
      weights_path, classes_list, (img_height, img_width))

  video_path = 'static/uploaded.mp4'

  hf.save_frame_range_video(
      video_path, saving_video_file_name, start, end)

  # global background
  sorted_archive, background, initial_time, length = hf.motion_tracking(video_path,
                                                                        model,
                                                                        classes,
                                                                        video_name=video_name,
                                                                        file_name=file_name,
                                                                        skip_frame=10,
                                                                        min_dist_thresh=35,
                                                                        removing_thresh=10,
                                                                        confi_thresh=0.1,
                                                                        start_sec=start,
                                                                        end_sec=end)

  end_time = (initial_time + dt.timedelta(seconds=length)
              ).replace(microsecond=0)
  duration = str(dt.timedelta(seconds=length)).split('.')[0]

  clip_duration = end - start
  clip_start = initial_time + dt.timedelta(seconds=start)
  clip_end = initial_time + dt.timedelta(seconds=end)

  K.clear_session()

  # title = 'title sample'
  # initial_time = 22
  # end_time = 33
  # duration = 11

  return render_template("results.html",
                         saving_video_file_name=saving_video_file_name,
                         video_name=video_name,
                         file_name=file_name,
                         initial_time=initial_time,
                         end_time=end_time,
                         duration=duration,
                         clip_start_time=clip_start,
                         clip_end_time=clip_end,
                         clip_duration=clip_duration,
                         my_form_result="processed")


@app.route('/display', methods=["GET", "POST"])  # to display.html
def KYM_display():
  return render_template("display.html")
  # initial_time=initial_time)


def to_datetime(time):
  month = int(time.split()[0].split('/')[0])
  day = int(time.split()[0].split('/')[1])
  year = int(time.split()[0].split('/')[2])
  hour = int(time.split()[1].split(':')[0])
  minute = int(time.split()[1].split(':')[1])
  if time.split()[-1] == 'PM':
    hour += 12
  return dt.datetime(year, month, day, hour, minute)


@app.route('/images', methods=["GET", "POST"])  # to display.html
def KYM_images():
  contour_img_name = 'img_contour_map'
  trajectory_img_name = 'img_traj_map'

  input_start_filter = request.args.get('input_start_filter')
  input_end_filter = request.args.get('input_end_filter')
  input_flag = request.args.get('input_flag')

  input_start_filter = to_datetime(input_start_filter)

  sorted_archive = pickle.load(
      open(file_name + '.pkl', 'rb'))
  background = pickle.load(open('background.pkl', 'rb'))

  if input_end_filter == '':
    sliced_archive = hf.time_slice(
        sorted_archive, dt_obj=[input_start_filter, input_end_filter], flag=input_flag)

    if sliced_archive == 'wrong method!':
      my_form_result = 'error'
      result = sliced_archive
    elif not sliced_archive:
      my_form_result = 'error'
      result = 'no record exist!'
    else:
      hf.contour_draw(sliced_archive, background, img_name=contour_img_name, alpha=0.6,
                      n_levels=5, figsize=(10, 5))
      hf.tracjactory_draw(sliced_archive, background, img_name=trajectory_img_name, alpha=0.3,
                          markersize=20, lw=10, figsize=(10, 5))
      my_form_result = 'filtered'
      result = None

  else:
    input_end_filter = to_datetime(input_end_filter)

    if input_start_filter >= input_end_filter:
      my_form_result = 'error'
      result = 'start time must be prior to the end time'
    else:
      sliced_archive = hf.time_slice(
          sorted_archive, dt_obj=[input_start_filter, input_end_filter], flag=input_flag)

      if sliced_archive == 'wrong method!':
        my_form_result = 'error'
        result = sliced_archive
      elif not sliced_archive:
        my_form_result = 'error'
        result = 'no record exist!'
      else:
        hf.contour_draw(sliced_archive, background, img_name=contour_img_name, alpha=0.6,
                        n_levels=5, figsize=(10, 5))
        hf.tracjactory_draw(sliced_archive, background, img_name=trajectory_img_name, alpha=0.3,
                            markersize=20, lw=10, figsize=(10, 5))
        my_form_result = 'filtered'
        result = None

  return render_template("display.html",
                         result=result,
                         contour_img_name=contour_img_name + '.png',
                         trajectory_img_name=trajectory_img_name + '.png',
                         my_form_result=my_form_result)


# start the server with the 'run()' method
if __name__ == "__main__":

  # will run locally http://127.0.0.1:5000/
  app.run(debug=True)

  # app.run(host="0.0.0.0", debug=True)
