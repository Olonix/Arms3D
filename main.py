from Full_Body import frame_processing_full_body, motion_filtering, filter_kalman, count_metrics_full_body
import mkv2frames
import make_anim

def main():
    mkv2frames.make_frames()
    frame_processing_full_body.main_processor()

    try:
        motion_filtering.process_skeletons()
    except IndexError as e:
        print(f"WARNING: An error occurred while filtering skeletons: {e}.")

    filter_kalman.apply_ekf()
    count_metrics_full_body.leg_movement_analysis()
    make_anim.make_animation()

if __name__ == "__main__":
    main()
