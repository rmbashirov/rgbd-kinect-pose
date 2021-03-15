from contextlib import contextmanager
import numpy as np
from threading import Thread, Lock

import torch
import pycuda.driver
from pycuda.gl import graphics_map_flags
from glumpy import app, gloo, gl


@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()


def create_shared_texture(w, h, c=4,
        map_flags=graphics_map_flags.WRITE_DISCARD,
        dtype=np.uint8):
    """Create and return a Texture2D with gloo and pycuda views."""
    tex = np.zeros((h,w,c), dtype).view(gloo.Texture2D)
    tex.activate() # force gloo to create on GPU
    tex.deactivate()
    cuda_buffer = pycuda.gl.RegisteredImage(int(tex.handle), tex.target, map_flags)
    return tex, cuda_buffer


def get_screen_program(tex):
    vertex = """
        uniform float scale;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            v_texcoord = texcoord;
            gl_Position = vec4(scale*position, 0.0, 1.0);
        } """
    fragment = """
        uniform sampler2D tex;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(tex, v_texcoord);
        } """
    # Build the program and corresponding buffers (with 4 vertices)
    screen = gloo.Program(vertex, fragment, count=4)
    # Upload data into GPU
    screen['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    screen['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    screen['scale'] = 1.0
    screen['tex'] = tex
    return screen


class CudaViewerThread(Thread):
    def __init__(self, h, w, max_framerate=30):
        super().__init__()
        self.h = h
        self.w = w
        self.max_framerate = max_framerate
        self.lock = Lock()
        self.is_shown = True
        self.state = None

    def on_draw(self, dt):
        with self.lock:
            if not self.is_shown:
                tensor = self.state
                self.is_shown = True
            else:
                return

        tex = self.screen_program['tex']
        h, w = tex.shape[:2]

        assert tex.nbytes == tensor.numel() * tensor.element_size()
        with cuda_activate(self.cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(tensor.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes // h
            cpy.height = h
            cpy(aligned=False)
            torch.cuda.synchronize()
        # draw to screen
        self.window.clear()
        self.screen_program.draw(gl.GL_TRIANGLE_STRIP)

    def on_close(self):
        pycuda.gl.autoinit.context.pop()

    def update_state(self, state):
        """
        state of shape (self.h, self.w, 3), dtype=torch.uint8
        """
        assert len(state.shape) == 3
        assert state.shape[-1] == 3
        assert state.dtype == torch.uint8
        with self.lock:
            self.state = torch.cat((state, state[:, :, [0]]), 2)
            self.state[..., 3] = 255
            self.state = torch.flip(self.state, (0,)).contiguous()
            self.is_shown = False

    def run(self):
        app.use('glfw')
        self.window = app.Window(height=self.h, width=self.w, fullscreen=False)
        self.window.push_handlers(on_draw=self.on_draw)
        self.window.push_handlers(on_close=self.on_close)

        import pycuda.gl.autoinit
        import pycuda.gl

        tex, cuda_buffer = create_shared_texture(self.w, self.h, 4)
        self.tex = tex
        self.cuda_buffer = cuda_buffer
        self.screen_program = get_screen_program(tex)

        app.run(framerate=self.max_framerate)


class CudaViewerThread2(Thread):
    def __init__(self, h, w, queue, max_framerate=20):
        super().__init__()
        self.h = h
        self.w = w
        self.queue = queue
        self.max_framerate = max_framerate
        self.is_shown = True
        self.count = 0

        self.tex = None
        self.cuda_buffer = None
        self.screen_program = None

    def on_draw(self, dt):
        # self.count += 1
        # state = self.queue.get()
        # if self.count < 100:
        #     return
        #
        # assert len(state.shape) == 3
        # assert state.shape[-1] == 3
        # assert state.dtype == torch.uint8
        #
        # print('on draw')
        #
        # state = torch.cat((state, state[:, :, [0]]), 2)
        # state[..., 3] = 255
        # state = torch.flip(state, (0,)).contiguous()
        # print(state.device, state.dtype, state.shape)

        tex = self.screen_program['tex']
        h, w = tex.shape[:2]

        # tensor = state
        tensor = self.state2

        assert tex.nbytes == tensor.numel() * tensor.element_size()
        with cuda_activate(self.cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(tensor.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes // h
            cpy.height = h
            cpy(aligned=False)
            torch.cuda.synchronize()
        # draw to screen
        self.window.clear()
        self.screen_program.draw(gl.GL_TRIANGLE_STRIP)

    def on_close(self):
        pycuda.gl.autoinit.context.pop()

    def run(self):
        app.use('glfw')
        self.window = app.Window(height=self.h, width=self.w, fullscreen=False)
        self.window.push_handlers(on_draw=self.on_draw)
        self.window.push_handlers(on_close=self.on_close)

        self.state2 = torch.ones((self.h, self.w, 4), dtype=torch.uint8, device=torch.device('cuda:0')) * 128
        self.state2[:, :, 1] = 255

        import pycuda.gl.autoinit
        import pycuda.gl

        tex, cuda_buffer = create_shared_texture(self.w, self.h, 4)
        self.tex = tex
        self.cuda_buffer = cuda_buffer
        self.screen_program = get_screen_program(tex)

        app.run(framerate=self.max_framerate)
