from typing import List, Optional, Protocol


class Recorder(Protocol):
    """Protocol defining required recorder interface"""

    @property
    def episode_id(self) -> int: ...
    @property
    def is_recording(self) -> bool: ...
    def start_streaming(self) -> bool: ...
    def stop_streaming(self) -> bool: ...
    def start_recording(self, episode_id: Optional[int] = None) -> bool: ...
    def stop_recording(self) -> bool: ...
    def reset_episode_recording(self) -> bool: ...
    def save_recordings(self) -> None: ...
    def peek_latest_frame(self, idx: int): ...
    def set_episode_id(self, episode_id: int) -> None: ...
    def clear_recording(self) -> None: ...
    def get_last_k_frames_from_all(self, k: int) -> dict: ...


class RecorderManager:
    def __init__(
        self,
        recorders: List[Recorder],
        verbose: bool = False,
    ):
        self.recorders = recorders
        self.verbose = verbose

    def set_episode_id(self, episode_id: int = None) -> None:
        for recorder in self.recorders:
            recorder.set_episode_id(episode_id)

    @property
    def episode_id(self) -> int:
        episode_ids = {recorder.episode_id for recorder in self.recorders}
        assert len(episode_ids) == 1, "All recorders should have the same episode ID"
        return episode_ids.pop()

    @property
    def is_recording(self) -> bool:
        """Check if all recorders are currently recording.

        Returns:
            bool: True if all recorders are recording, False if any are not recording.
        """
        return all(recorder.is_recording for recorder in self.recorders)

    def start_streaming(self) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.start_streaming()
        return success

    def stop_streaming(self) -> None:
        success = True
        for recorder in self.recorders:
            success &= recorder.stop_streaming()
        return success

    def reset_episode_recording(self) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.reset_episode_recording()
        return success

    def start_recording(self, episode_id: Optional[int] = None) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.start_recording(episode_id=episode_id)

        if success:
            if self.verbose:
                print(f"Started recording episode {self.episode_id}")
        return success

    def stop_recording(self) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.stop_recording()

        return success

    def clear_recording(self) -> None:
        for recorder in self.recorders:
            recorder.clear_recording()

    def save_recordings(self) -> None:
        for recorder in self.recorders:
            recorder.save_recordings()

    def get_latest_frames(self) -> dict:
        frames = {}
        for recorder in self.recorders:
            frames.update(recorder.get_last_k_frames_from_all(k=1))
        return frames
