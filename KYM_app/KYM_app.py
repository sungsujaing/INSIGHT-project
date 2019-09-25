from flask import Flask, render_template, request
import helper_func_app as hf

# Create the application object
app = Flask(__name__)


# we are now using these methods to get user input
@app.route('/', methods=["GET", "POST"])
def home_page():
  return render_template('index.html')


@app.route('/output')
def KYM_output():
  #
  # Pull input
  input_URL = request.args.get('input_URL')
  input_start = request.args.get('input_start')
  input_end = request.args.get('input_end')

  # Case if empty
  if input_URL == '':
    return render_template("index.html",
                           my_form_result="Empty")
  else:
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

    url = input_URL
    saving_video_file_name = 'original_video_clip'
    video_name = 'processed_video_clip'
    file_name = 'processed_file'

    hf.save_frame_range_video(
        url, saving_video_file_name, start, end)  # working fine!
    sorted_archive, background = hf.motion_tracking(url,                  # working fine!
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

    # sorted_archive = pickle.load(open('file_name' + '.pkl', 'rb'))

    # tm_1 = dt.datetime(2019, 9, 24, 9, 53, 44)
    # tm_2 = dt.datetime(2019, 9, 24, 9, 53, 49)

    # sliced_archive = hf.time_slice(sorted_archive, dt_obj=[tm_1, tm_2], flag='in')

    hf.contour_draw(sorted_archive, background, alpha=0.6,
                    n_levels=5, figsize=(10, 5))
    hf.tracjactory_draw(sorted_archive, background, alpha=0.3,
                        markersize=20, lw=10, figsize=(10, 5))

    some_output = 3
    some_number = 2
    # some_image = "gifpy.gif"
    return render_template("results.html",
                           my_input=type(end),
                           my_output=input_URL,
                           my_number=end,
                           my_img_name=some_output,
                           my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
  app.run(debug=True)  # will run locally http://127.0.0.1:5000/
