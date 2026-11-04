"""Reusable recording lifecycle, independent of keyboard and camera hardware."""
from pathlib import Path


class RecordingSession:
    def __init__(self, manager):
        self.manager = manager

    def __enter__(self):
        if not self.manager.start_streaming():
            self.manager.stop_streaming()
            raise RuntimeError('Could not start camera streams')
        return self

    def handle_key(self, key):
        """s=start, w=save, a=discard/restart, q=quit; return False to exit."""
        if key == 's' and self.manager.reset_episode_recording():
            self.manager.start_recording()
        elif key == 'w' and self.manager.stop_recording():
            self.manager.save_recordings()
        elif key in ('a', 'q'):
            if self.manager.stop_recording():
                self.manager.clear_recording()
            if key == 'a' and self.manager.reset_episode_recording():
                self.manager.start_recording()
            return key != 'q'
        return True

    def __exit__(self, *exc):
        try:
            self.manager.stop_recording()
        finally:
            self.manager.stop_streaming()


def create_session(output_dir, *, backend='iphone-gopro', fps=30, device_index=0,
                   video_path=None, verbose=False, cameras=None):
    from .camera.camera import FrameData
    from .recorder.manager import RecorderManager
    from .recorder.video import VideoRecorder
    if fps <= 0:
        raise ValueError('fps must be positive')
    if cameras is None:
        from .camera.iphone import IPhone
        cameras = [IPhone()]
        if backend == 'iphone-gopro':
            from .camera.gopro import GoPro
            cameras.append(GoPro(fps=fps, device_index=device_index, video_path=video_path))
        elif backend == 'iphone-oak':
            from .camera.oak import Oak, get_all_oak_cameras
            devices = get_all_oak_cameras()
            if not devices:
                raise RuntimeError('No OAK camera connected')
            cameras.append(Oak('OakCamera', fps=fps, device_id=devices[0]))
        elif backend != 'iphone':
            raise ValueError(f'Unknown recording backend: {backend}')
    recorder = VideoRecorder(record_fps=fps, stream_fps=fps,
        video_record_path=str(Path(output_dir).expanduser()), camera_sources=cameras,
        frame_data_class=FrameData, verbose=verbose)
    return RecordingSession(RecorderManager([recorder], verbose=verbose))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Record synchronized iPhone and GoPro/OAK streams')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--backend', choices=['iphone-gopro', 'iphone-oak', 'iphone'], default='iphone-gopro')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--gopro-device', type=int, default=0)
    parser.add_argument('--gopro-video')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    import cv2
    from spvtp.common.frame_manager import FrameRateContext
    session = create_session(args.data_dir, backend=args.backend, fps=args.fps,
        device_index=args.gopro_device, video_path=args.gopro_video, verbose=args.verbose)
    print('s=start | w=save | a=restart | q=quit')
    try:
        with session:
            while True:
                with FrameRateContext(args.fps):
                    for name, frames in session.manager.get_latest_frames().items():
                        if frames and frames[-1] is not None and frames[-1].rgb is not None:
                            cv2.imshow(name, frames[-1].rgb)
                    key = cv2.waitKey(1) & 0xff
                    if key != 255 and not session.handle_key(chr(key)):
                        break
    finally:
        cv2.destroyAllWindows()
