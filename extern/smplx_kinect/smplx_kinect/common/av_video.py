# av installation:
# using conda:
#   RUN conda install av -c conda-forge
# or using pip (ubuntu 18 required):
#     RUN apt-get install -y software-properties-common && \
#         add-apt-repository ppa:djcj/hybrid && \
#         apt-get update && \
#         apt-get install -y ffmpeg
#
#     RUN apt-get update && apt-get install -y \
#         pkg-config \
#         libavformat-dev \
#         libavcodec-dev \
#         libavdevice-dev \
#         libavutil-dev \
#         libswscale-dev \
#         libswresample-dev \
#         libavfilter-dev
#     RUN pip install av


import av
import os
from fractions import Fraction


class VideoReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.container = av.open(self.filepath)
        self.stream = None
        for stream in self.container.streams:
            if stream.type == 'video':
                if self.stream is not None:
                    print('More than 1 video streams found')
                self.stream = stream
        assert self.stream is not None, 'no video stream'
        self.codec_context = self.stream.codec_context
        self.current_frame_idx = -1  # first frame idx = 0

    def frames_iterator(self, vsync=0, verbose_every=None, return_time=False):
        if verbose_every is not None:
            print('Starting to read {} frames'.format(self.frames()))
        for frame in self.container.decode(video=vsync):
            rgb24_img = frame.to_ndarray(format='rgb24')
            self.current_frame_idx += 1
            if verbose_every is not None:
                if self.current_frame_idx % verbose_every == 0:
                    print('read {} frames'.format(self.current_frame_idx))
            if return_time:
                result = rgb24_img, frame.time
            else:
                result = rgb24_img
            yield result

    def current_frame_idx(self):
        return self.current_frame_idx

    def height(self):
        return self.codec_context.coded_height

    def width(self):
        return self.codec_context.coded_width

    def codec_name(self):
        return self.codec_context.codec.long_name

    def frames(self):
        return self.stream.frames

    def fps(self):
        return float(self.codec_context.framerate)

    def close(self):
        self.container.close()


class VideoWriter:
    def __init__(self, filepath, fps, codec='h264', br_scale=1):
        '''
        :param codec: you may try another codec: 'mpeg4'
        :param br_scale: you may scale bit rate for better video quality or lower video size
        '''
        self.filepath = filepath
        assert os.path.splitext(self.filepath)[1] == '.mp4'
        self.container = av.open(self.filepath, mode='w')
        self.fps = Fraction(fps).limit_denominator(2 ** 16 - 1)  # 65535
        self.codec = codec
        self.stream = self.container.add_stream(self.codec, self.fps)
        self.current_frame_idx = -1
        self.configured = False
        self.br_scale = br_scale

    def add_frame(self, img):
        if not self.configured:
            h, w = img.shape[:2]
            if self.codec == 'h264':
                h += h % 2
                w += w % 2
            max_br = 2 ** 31 - 1  # 2147483647
            br = min(int(h * w * 10 * self.br_scale), max_br)
            self.stream.bit_rate = br
            self.stream.bit_rate_tolerance = br // 20
            self.stream.height = h
            self.stream.width = w
            self.configured = True

        frame = av.VideoFrame.from_ndarray(img, format='rgb24')

        packet = self.stream.encode(frame)
        self.container.mux(packet)
        self.current_frame_idx += 1

    def current_frame_idx(self):
        return self.current_frame_idx

    def close(self):
        packet = self.stream.encode(None)
        self.container.mux(packet)
        self.container.close()