from roboverse.utils.camera import Camera
import cv2
cam_width, cam_height = 1920, 1080
cam_src = 0
cam_fps = 30

if __name__ == "__main__":
    cam = Camera(cam_src, cam_width, cam_height, cam_fps)
    while True:
        frame = cam.get_frame()
        if frame is not None:
            print('Got frame')
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.stop_cam()