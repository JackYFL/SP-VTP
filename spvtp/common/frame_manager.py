import time


class FrameRateContext:
    def __init__(self, frame_rate, verbose=False):
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
        self.verbose = verbose
        self.start_time = time.monotonic()
        self.iter = 0
        # Compensation for sleep overhead
        self.sleep_overhead = 0.000  # Initial estimate, will be adjusted
        self.alpha = 0.0  # Learning rate for sleep overhead adjustment

    def __enter__(self):
        self.frame_start_time = time.monotonic()
        self.this_iter_end_time = self.start_time + (self.iter + 1) * self.dt
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        now = time.monotonic()

        if now < self.this_iter_end_time:
            target_sleep = self.this_iter_end_time - now
            # Compensate for sleep overhead
            adjusted_sleep = max(0, target_sleep - self.sleep_overhead)

            if self.verbose:
                print(
                    f"Target sleep: {target_sleep:.6f}, Adjusted sleep: {adjusted_sleep:.6f}"
                )

            sleep_start = time.monotonic()
            time.sleep(adjusted_sleep)
            actual_sleep = time.monotonic() - sleep_start

            # Update sleep overhead estimate
            if actual_sleep > 0:
                # measured_overhead = actual_sleep - adjusted_sleep
                self.sleep_overhead = (
                    1 - self.alpha
                ) * self.sleep_overhead + self.alpha * actual_sleep
        else:
            if self.verbose:
                print("Frame took longer than expected")

        self.iter += 1

        if self.verbose:
            frame_time = time.monotonic() - self.frame_start_time
            current_fps = 1.0 / frame_time
            print("iteration:", self.iter)
            print(
                f"Frame time: {frame_time:.6f}s, Current FPS: {current_fps:.1f}, Sleep overhead: {self.sleep_overhead:.6f}"
            )


def precise_sleep(dt: float, slack_time: float = 0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return
