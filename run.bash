# conda activate dsec

# python ./scripts/events_to_slice.py /home/zzt/Project/EVGGT/interlaken_00_c/events/left/events.h5 /home/zzt/Project/EVGGT/DSEC/output

# python ./scripts/trans_disparity_2_depth.py --disparity_path /home/zzt/Project/EVGGT/interlaken_00_c/disparity/event/000000.png \
#                  --calibration_path /home/zzt/Project/EVGGT/interlaken_00_c/calibration/cam_to_cam.yaml \
#                  --output_path /home/zzt/Project/EVGGT/interlaken_00_c/disparity/frame_depth.png

# python ./scripts/trans_dis2dep_single_folder.py \
#     --disparity_dir /home/zzt/Project/EVGGT/interlaken_00_c/disparity/event \
#     --calibration_path /home/zzt/Project/EVGGT/interlaken_00_c/calibration/cam_to_cam.yaml \
#     --output_dir /home/zzt/Project/EVGGT/interlaken_00_c/depth                 

python ./scripts/trans_dis2dep_dataset.py \
    --disparity_dir /home/zzt/data/dsec/train \
    --output_dir /home/zzt/data/dsec/train