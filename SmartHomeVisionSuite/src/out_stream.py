import ffmpeg


class RTSPFrameStreamer:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.ffmpeg_container = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(740, 420), framerate=5)
            .output(
                rtsp_url,
                vcodec='libx264',
                preset='ultrafast',
                pix_fmt='yuvj420p',
                r=5,
                f='rtsp',
                rtsp_transport='tcp',
            )
            .global_args('-hide_banner', '-loglevel', 'info')
            .run_async(pipe_stdin=True)
        )

    def write_frame(self, frame):
        try:
            self.ffmpeg_container.stdin.write(frame)
        except BrokenPipeError:
            pass

    def close(self):
        self.ffmpeg_container.stdin.close()
        self.ffmpeg_container.wait()
