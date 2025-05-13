import taichi as ti
import taichi.math as tm

from tinycio.fsio import load_image
from tinytex.ti import Texture2D, Sampler2D, WrapMode, FilterMode

ti.init(arch=ti.vulkan)

res = (512, 512)
pixels = ti.Vector.field(3, dtype=float, shape=res)

@ti.data_oriented
class TextureSamplingExample:
    def __init__(self, tex:Texture2D):        
        self.tex = tex
        wrap_mode = WrapMode.REPEAT
        self.s_bilin = Sampler2D(wrap_mode=wrap_mode, filter_mode=FilterMode.BILINEAR)
        self.s_bicub = Sampler2D(wrap_mode=wrap_mode, filter_mode=FilterMode.HERMITE)
        self.s_mitne = Sampler2D(wrap_mode=wrap_mode, filter_mode=FilterMode.MITCHELL_NETRAVALI)
        self.s_bspln = Sampler2D(wrap_mode=wrap_mode, filter_mode=FilterMode.B_SPLINE)

    @ti.kernel
    def paint(self, t:ti.f32):
        for i, j in pixels:
            uv = ti.Vector([i / res[0], 1. - (j / res[1])])
            warp_uv = uv + ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.002
            warp_uv *= 4.
            c = tm.vec4(0.0)
            if uv.x > 0.3:
                if uv.y < 0.25:     c = self.s_bilin.sample_lod(self.tex, warp_uv, (tm.sin(t * 0.15) + 1) * 3.)
                elif uv.y < 0.5:    c = self.s_bicub.sample_lod(self.tex, warp_uv, (tm.sin(t * 0.15) + 1) * 3.)
                elif uv.y < 0.75:   c = self.s_mitne.sample_lod(self.tex, warp_uv, (tm.sin(t * 0.15) + 1) * 3.)
                else:               c = self.s_bspln.sample_lod(self.tex, warp_uv, (tm.sin(t * 0.15) + 1) * 3.)
            else: 
                c = self.s_bilin.fetch(self.tex, tm.ivec2(i, res[1] - j))

            bg = tm.vec3(tm.cos(t * 0.05) * 0.5 + 0.5, tm.sin(t * 0.05) * 0.5 + 0.5, 1.) * 0.25
            c.rgb = tm.mix(bg.rgb, c.rgb, c.a)
            pixels[i, j] = [c.r, c.g, c.b]

def main():
    im = load_image('../doc/images/waits_color.png')
    tex = Texture2D(im, generate_mips=True)
    example = TextureSamplingExample(tex)
    
    window = ti.ui.Window("Taichi Texture Sampling", res, fps_limit=60)
    canvas = window.get_canvas()
    t = 0.0
    while window.running:
        example.paint(t)
        canvas.set_image(pixels)
        window.show()
        t += 0.15

if __name__ == "__main__":
    main()