import helper_func_app as hf

img_height = 512
img_width = 512
weights_path = 'VGG_VOC0712_SSD_512x512_iter_120000.h5'
classes_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

model, classes, classes_rev = hf.build_SSD(
    weights_path, classes_list, (img_height, img_width))

url = 'https://www.youtube.com/watch?v=KMJS66jBtVQ'
saving_video_file_name = 'original_video_clip'
video_name = 'processed_video_clip'
file_name = 'processed_file'
start = 0
end = 5

hf.save_frame_range_video(url, saving_video_file_name, start, end)
print('')
sorted_archive, background = hf.motion_tracking(url,
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

# sorted_archive = pickle.load(open('file_name'+'.pkl','rb'))

# tm_1 = dt.datetime(2019, 9, 24, 9, 53, 44)
# tm_2 = dt.datetime(2019, 9, 24, 9, 53, 49)

# sliced_archive = hf.time_slice(sorted_archive, dt_obj=[tm_1, tm_2], flag='in')

hf.contour_draw(sorted_archive, background, alpha=0.6,
                n_levels=5, figsize=(13, 6))
hf.tracjactory_draw(sorted_archive, background, alpha=0.3,
                    markersize=20, lw=10, figsize=(10, 5))
