import threading
import head_pose
import detection

if __name__ == "__main__":
    # Head Pose runs in background thread
    t1 = threading.Thread(target=head_pose.run_head_pose)

    t1.start()

    # Matplotlib MUST run on main thread
    detection.run_detection()

    t1.join()
