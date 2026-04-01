import functools
from typing import Optional
import numpy as np
import panel as pn
import skimage.transform as sktransform
from skimage.transform import AffineTransform

from bokeh.plotting import figure
from bokeh.models.widgets import CheckboxGroup, Spinner, Slider
from bokeh.models import CustomJS
from bokeh.models import Scatter, Image
from bokeh.models import ColumnDataSource
from bokeh.models.tools import PointDrawTool
from bokeh.palettes import Blues256, Reds256

pn.extension()


class ImageTransformer:
    """
    Provides method to compute image transformations
    Supports chaining multiple transforms and changes to output shape
    Provides the final transformation matrix via the
    :code:`get_combined_transform()` method

    Based entirely on `skimage.transform`
    """
    def __init__(self, image):
        self._image = image
        self._transforms = []
        self._reshapes = []
        self._frozen_len = -1

    def set_image(self, image):
        self._image = image

    @property
    def transforms(self):
        return self._transforms

    def add_transform(self, *transforms, output_shape=None, frozen=False):
        self.transforms.append(self._combine_transforms(*transforms))
        self._reshapes.append(output_shape)
        if frozen:
            self._frozen_len = len(self.transforms)

    def add_null_transform(self, output_shape=None, frozen=False):
        self.transforms.append(self._null_transform())
        self._reshapes.append(output_shape)
        if frozen:
            self._frozen_len = len(self.transforms)

    def remove_transform(self, n=1):
        for _ in range(n):
            if len(self.transforms) <= self._frozen_len:
                break
            try:
                self.transforms.pop(-1)
                self._reshapes.pop(-1)
            except (IndexError, TypeError):
                break

    @staticmethod
    def _null_transform():
        return sktransform.EuclideanTransform()

    def clear_transforms(self):
        self.transforms.clear()
        self._reshapes.clear()

    def current_shape(self):
        reshapes = [r for r in self._reshapes if r is not None]
        if reshapes:
            return reshapes[-1]
        return self._image.shape[:2]

    def get_combined_transform(self):
        transform_mxs = [t.params for t in self.transforms]
        if not transform_mxs:
            transform_mat = self._null_transform().params
        elif len(transform_mxs) >= 2:
            transform_mat = functools.reduce(np.matmul, transform_mxs)
        else:
            transform_mat = transform_mxs[0]
        combined = sktransform.AffineTransform(matrix=transform_mat)
        return combined

    @staticmethod
    def _combine_transforms(*transforms):
        transforms = [sktransform.AffineTransform(matrix=t)
                      if isinstance(t, np.ndarray)
                      else t for t in transforms]
        if not transforms:
            return ImageTransformer._null_transform()
        elif len(transforms) == 1:
            return transforms[0]
        else:
            transform_mat = functools.reduce(np.matmul, [t.params for t in transforms])
            return sktransform.AffineTransform(matrix=transform_mat)

    def get_transformed_image(self, preserve_range=True, order=None, cval=np.nan, **kwargs):
        if not self.transforms:
            return self._image
        combined_transform = self.get_combined_transform()
        return sktransform.warp(self._image,
                                combined_transform,
                                order=order,
                                output_shape=kwargs.pop('output_shape', self.current_shape()),
                                preserve_range=preserve_range,
                                cval=cval,
                                **kwargs)

    def get_current_center(self):
        current_shape = np.asarray(self.current_shape())
        return current_shape / 2.

    def translate(self, xshift=0., yshift=0., output_shape=None):
        transform = sktransform.EuclideanTransform(translation=(xshift, yshift))
        self.add_transform(transform, output_shape=output_shape)

    def shear(self, xshear=0., yshear=0., output_shape=None):
        transform = sktransform.AffineTransform(shear=(xshear, yshear))
        self.add_transform(transform, output_shape=output_shape)

    def rotate_about_point(self, point_yx, rotation_degrees=None, rotation_rad=None):
        if rotation_degrees and rotation_rad:
            raise ValueError('Cannot specify both degrees and radians')
        elif rotation_degrees:
            rotation_rad = np.deg2rad(rotation_degrees)
        if not rotation_rad:
            if rotation_rad == 0.:
                return
            raise ValueError('Must specify one of degrees or radians')

        transform = sktransform.EuclideanTransform(rotation=rotation_rad)
        self._operation_with_origin(point_yx, transform)

    def rotate_about_center(self, **kwargs):
        current_center = self.get_current_center()
        return self.rotate_about_point(current_center, **kwargs)

    def uniform_scale_centered(self, scale_factor, output_shape=None):
        transform = sktransform.SimilarityTransform(scale=scale_factor)
        current_center = self.get_current_center()
        self._operation_with_origin(current_center, transform)

    def xy_scale_about_point(self, point_yx, xscale=1., yscale=1.):
        transform = sktransform.AffineTransform(scale=(xscale, yscale))
        self._operation_with_origin(point_yx, transform)

    def xy_scale_about_center(self, **kwargs):
        current_center = self.get_current_center()
        return self.xy_scale_about_point(current_center, **kwargs)

    def _operation_with_origin(self, origin_yx, transform):
        origin_xy = np.asarray(origin_yx).astype(float)[::-1]
        forward_shift = sktransform.EuclideanTransform(translation=origin_xy)
        backward_shift = sktransform.EuclideanTransform(translation=-1 * origin_xy)
        self.add_transform(forward_shift, transform, backward_shift)

    @staticmethod
    def available_transforms():
        """The transformation types which can be estimated, compatible with this class"""
        return ['affine', 'euclidean', 'similarity', 'projective']

    def estimate_transform(self, static_points, moving_points,
                           method='affine', output_shape=None, clear=False):
        assert method in self.available_transforms()
        assert static_points.size and moving_points.size, 'Need points to match'
        assert static_points.size == moving_points.size, 'Must supply matching pointsets'
        transform = sktransform.estimate_transform(method,
                                                   static_points.reshape(-1, 2),
                                                   moving_points.reshape(-1, 2))
        if clear:
            self.clear_transforms()
        self.add_transform(transform, output_shape=output_shape)
        return transform


def set_frame_height(fig, shape, maxdim=450, mindim=-1):
    h, w = shape
    if h > w:
        fh = maxdim
        fw = max(mindim, int((w / h) * maxdim))
    else:
        fw = maxdim
        fh = max(mindim, int((h / w) * maxdim))
    fig.frame_height = fh
    fig.frame_width = fw
    return fh, fw


def adapt_figure(fig: figure, shape, maxdim: int | None = 450, mindim: int | None = None):
    if mindim is None:
        # Don't change aspect ratio in this case
        mindim = -1

    fig.y_range.flipped = True
    fh, fw = set_frame_height(fig, shape, maxdim, mindim)

    if fh > 0.8 * fw:
        location = 'right'
    else:
        location = 'below'
    fig.toolbar_location = location

    fig.x_range.range_padding = 0.
    fig.y_range.range_padding = 0.
    fig.toolbar.active_drag = None
    fig.background_fill_alpha = 0.
    fig.border_fill_color = None
    # zoom_tools = tuple(t for t in fig.tools if isinstance(t, WheelZoomTool))
    # try:
    #     fig.toolbar.active_scroll = zoom_tools[0]
    # except IndexError:
    #     pass


def format_transform_md(transform: AffineTransform):
    transform = AffineTransform(matrix=transform.params)
    scale_x, scale_y = transform.scale
    trans_x, trans_y = transform.translation
    return f"""| Rotation | Scale   | Shear    | Translation   |
| -------- | ------- | -------- | -------       |
| {np.rad2deg(transform.rotation):.1f}  | ({scale_x:.3f}, {scale_y:.3f}) | {transform.shear:.2f}  | ({trans_x:.1f}, {trans_y:.1f}) |
"""  # noqa


# Unicode arrow codes used for defining UI buttons
LEFT_ARROW = '\u25C1'
UP_ARROW = '\u25B3'
RIGHT_ARROW = '\u25B7'
DOWN_ARROW = '\u25BD'
ROTATE_RIGHT_ARROW = '\u21B7'
ROTATE_LEFT_ARROW = '\u21B6'
SHEAR_MORE = '+'
SHEAR_LESS = '-'


def translate_buttons(cb, width: int = 40, height: int = 40, margin: tuple[int, int] = (2, 2)):
    """
    A button array for up/down/left/right
    Configured for y-axis pointing down!!
    """
    kwargs = {
        'width': width,
        'height': height,
        'margin': margin,
        'sizing_mode': 'fixed',
    }
    get_sp = lambda: pn.Spacer(**kwargs)  # noqa
    button_kwargs = {
        'button_type': 'primary',
        **kwargs,
    }
    left = pn.widgets.Button(name=LEFT_ARROW, **button_kwargs)
    left.on_click(functools.partial(cb, x=-1))
    up = pn.widgets.Button(name=UP_ARROW, **button_kwargs)
    up.on_click(functools.partial(cb, y=-1))
    right = pn.widgets.Button(name=RIGHT_ARROW, **button_kwargs)
    right.on_click(functools.partial(cb, x=1))
    down = pn.widgets.Button(name=DOWN_ARROW, **button_kwargs)
    down.on_click(functools.partial(cb, y=1))
    return pn.Column(
        pn.Row(get_sp(), up, get_sp(), margin=(0, 0)),
        pn.Row(left, down, right, margin=(0, 0)),
        # pn.Row(get_sp(), down, get_sp(), margin=(0, 0)),
        margin=(0, 0),
    )


def shear_buttons(cb, width: int = 40, height: int = 40, margin: tuple[int, int] = (2, 2)):
    """
    Buttons for x and y shear
    """
    kwargs = {
        'width': width,
        'height': height,
        'margin': margin,
        'sizing_mode': 'fixed',
    }
    get_sp = lambda: pn.Spacer(**kwargs)  # noqa
    button_kwargs = {
        'button_type': 'primary',
        **kwargs,
    }
    left = pn.widgets.Button(name=SHEAR_LESS, **button_kwargs)
    left.on_click(functools.partial(cb, x=-1))
    up = pn.widgets.Button(name=SHEAR_MORE, **button_kwargs)
    up.on_click(functools.partial(cb, y=-1))
    right = pn.widgets.Button(name=SHEAR_MORE, **button_kwargs)
    right.on_click(functools.partial(cb, x=1))
    down = pn.widgets.Button(name=SHEAR_LESS, **button_kwargs)
    down.on_click(functools.partial(cb, y=1))
    return pn.Column(
        pn.Row(get_sp(), up, get_sp(), margin=(0, 0)),
        pn.Row(left, down, right, margin=(0, 0)),
        # pn.Row(get_sp(), down, get_sp(), margin=(0, 0)),
        margin=(0, 0),
    )


def rotate_buttons(cb):
    """A button array for rotate acw / cw"""
    width = height = 40
    margin = (2, 2)
    sp = pn.Spacer(width=width, height=height, margin=margin)
    kwargs = {'width': width, 'height': height, 'margin': margin, 'button_type': 'primary'}
    acw_btn = pn.widgets.Button(name=ROTATE_LEFT_ARROW, **kwargs)
    acw_btn.on_click(functools.partial(cb, dir=-1))
    cw_btn = pn.widgets.Button(name=ROTATE_RIGHT_ARROW, **kwargs)
    cw_btn.on_click(functools.partial(cb, dir=1))
    return pn.Row(sp, acw_btn, cw_btn, margin=(0, 0))


def scale_buttons(cb):
    """A button array for scaling x / y / xy up and down"""
    width = height = 40
    margin = (2, 2)
    text_kwargs = {'width': width // 2,
                   'height': height // 2,
                   'margin': margin,
                   'align': ('end', 'center')}
    button_kwargs = {'width': width,
                     'height': height,
                     'margin': margin,
                     'button_type': 'primary'}
    x_row = up_down_pair('X:',
                         cb,
                         {'xdir': 1},
                         {'xdir': -1},
                         text_kwargs,
                         button_kwargs)
    y_row = up_down_pair('Y:',
                         cb,
                         {'ydir': 1},
                         {'ydir': -1},
                         text_kwargs,
                         button_kwargs)
    xy_row = up_down_pair('XY:',
                          cb,
                          {'xdir': 1, 'ydir': 1},
                          {'xdir': -1, 'ydir': -1},
                          text_kwargs,
                          button_kwargs)
    lo = pn.Column(x_row,
                   y_row,
                   xy_row, margin=(0, 0))
    return lo


def up_down_pair(name, cb, upkwargs, downkwargs, text_kwargs, button_kwargs):
    sp = pn.Spacer(**text_kwargs)
    text = pn.widgets.StaticText(value=name, **text_kwargs)
    compress = pn.widgets.Button(name=f'{RIGHT_ARROW} {LEFT_ARROW}', **button_kwargs)
    compress.on_click(functools.partial(cb, **downkwargs))
    expand = pn.widgets.Button(name=f'{LEFT_ARROW} {RIGHT_ARROW}', **button_kwargs)
    expand.on_click(functools.partial(cb, **upkwargs))
    return pn.Row(sp, text, compress, expand, margin=(0, 0))


def _create_image(fig, array, palette='Viridis256'):
    glyph = Image(
        image='image', x='x', y='y', dw='dw', dh='dh'
    )
    height, width = array.shape
    cds_dict = {
        'x': [0 - 0.5],
        'y': [0 - 0.5],
        'dw': [width],
        'dh': [height],
        'image': [array],
    }
    cds = ColumnDataSource(cds_dict)
    fig.add_glyph(cds, glyph)
    return glyph, cds


def fine_adjust(
    static: np.ndarray,
    moving: np.ndarray,
    initial_transform: Optional['AffineTransform'] = None
):
    """
    Provides a UI panel to manually align the image moving onto static
    Optionally provide a skimage.transform.GeometricTransform object
    to pre-transform moving
    """
    transformer_moving = ImageTransformer(moving)
    if initial_transform:
        transformer_moving.add_transform(initial_transform, output_shape=static.shape)
    else:
        # To be sure we set the output shape to match static
        transformer_moving.add_null_transform(output_shape=static.shape)

    # static_name = 'Static'
    moving_name = 'Moving'

    fig = figure()
    fig_lo = pn.pane.Bokeh(fig)
    adapt_figure(fig, static.shape)

    static_im, _ = _create_image(fig, static)
    moving_im, moving_cds = _create_image(
        fig, transformer_moving.get_transformed_image(cval=np.nan)
    )

    static_im.color_mapper.palette = Reds256
    moving_im.color_mapper.palette = Blues256
    static_im.color_mapper.nan_color = (0,) * 4
    moving_im.color_mapper.nan_color = (0,) * 4
    moving_im.global_alpha = 0.5

    overlay_alpha = Slider(
        title=f'{moving_name} alpha',
        start=0.,
        end=1.,
        value=moving_im.global_alpha,
        step=0.01,
        syncable=False,
        max_width=200,
    )
    alpha_callback = CustomJS(
        args={'glyph': moving_im},
        code="""glyph.global_alpha = cb_obj.value;""",
    )
    overlay_alpha.js_on_change('value', alpha_callback)

    wobble_alpha_cbox = CheckboxGroup(
        labels=['Wobble alpha, step:'],
        active=[],
        align='center',
    )
    wobble_step_input = Spinner(
        low=0.05,
        high=0.25,
        value=0.1,
        step=0.05,
        align='end',
        width=100,
        format="0.2f",
    )
    wobble_callback = CustomJS(
        args=dict(
            alpha_slider=overlay_alpha,
            wobble_step=wobble_step_input,
        ),
        code=R'''
const TIMEOUT = 25
var RUNNING = false
var DIRECTION = 1

export default async function (args, obj, data, context) {
    const active = obj.active.length == 1
    if (!active) {
        RUNNING = false
        return
    }
    var stepsize = 0.05
    var current_val = args.alpha_slider.value

    RUNNING = true
    while (RUNNING) {
        current_val = args.alpha_slider.value
        stepsize = Math.min(Math.max(args.wobble_step.value, 0.01), 0.9)

        var new_val = Math.min(Math.max(current_val + DIRECTION * stepsize, 0), 1)
        if ((new_val >= 1) || (new_val <= 0)) {
            DIRECTION = -1 * DIRECTION
        }
        args.alpha_slider.value = new_val
        await new Promise(r => setTimeout(r, TIMEOUT));
    }
}
'''
    )
    wobble_alpha_cbox.js_on_change('active', wobble_callback)

    show_diff_cbox = pn.widgets.Checkbox(name='Show image difference',
                                         value=False,
                                         align='center')

    translate_step_input = pn.widgets.FloatInput(name='Translate step (px):',
                                                 value=1.,
                                                 start=0.1,
                                                 end=100.,
                                                 width=125)

    shear_step_input = pn.widgets.FloatInput(name='Shear step (deg):',
                                             value=1.,
                                             start=0.1,
                                             end=10.,
                                             width=125)

    def _transform_md():
        transform = transformer_moving.get_combined_transform()
        return format_transform_md(transform)

    transform_md = pn.pane.Markdown(
        object=_transform_md(),
    )

    def update_moving(*updates, fix_clims=True):
        moving = transformer_moving.get_transformed_image()
        if show_diff_cbox.value:
            to_display = np.float32(moving) - np.float32(static)
        else:
            to_display = moving
        moving_cds.data.update(
            dict(image=[to_display])
        )
        pn.io.push_notebook(fig_lo)
        transform_md.object = _transform_md()

    def switch_diff_image(event):
        if event.new:
            overlay_alpha.value = 1.
        else:
            overlay_alpha.value = 0.5
        update_moving()

    show_diff_cbox.param.watch(switch_diff_image, 'value')

    def fine_translate(event, x=0, y=0):
        if not x and not y:
            return
        raw_adjust = -1 * translate_step_input.value
        transformer_moving.translate(xshift=x * raw_adjust, yshift=y * raw_adjust)
        update_moving()

    def fine_shear(event, x=0, y=0):
        if not x and not y:
            return
        raw_adjust = (np.pi / 360.0 * shear_step_input.value)
        transformer_moving.shear(xshear=raw_adjust * x, yshear=raw_adjust * y)
        update_moving()

    _cy, _cx = tuple(a // 2 for a in static.shape)
    cursor_cds = ColumnDataSource(data=dict(x=[_cx], y=[_cy]))

    def _cursor_pos(cursor_cds):
        data = cursor_cds.data
        return data['x'][0], data['y'][0]

    cursor = Scatter(
        marker='circle_dot',
        line_color='cyan',
        line_width=2,
        line_alpha=0.,
        fill_alpha=0,
        size=15,
        hit_dilation=2.0,
    )
    cursor_ren = fig.add_glyph(cursor_cds, cursor)

    cursor_drag = PointDrawTool(
        name="Cursor",
        description="Drag the cursor",
        renderers=[cursor_ren],
        add=False,
        drag=True,
        num_objects=1,
        empty_value=1,
    )
    fig.add_tools(cursor_drag)
    # fig.toolbar.active_multi = cursor_drag

    about_center_cbox = pn.widgets.Checkbox(name='Center-origin',
                                            value=True,
                                            width=125)

    def _set_cursor_alpha(event):
        cursor.line_alpha = float(not event.new)

    about_center_cbox.param.watch(_set_cursor_alpha, 'value')

    rotate_step_input = pn.widgets.FloatInput(name='Rotate step (deg):',
                                              value=1.,
                                              start=0.1,
                                              end=100.,
                                              width=125)

    def fine_rotate(event, dir=0):
        if not dir:
            return
        about_center = about_center_cbox.value
        true_rotate = -1 * rotate_step_input.value * dir
        if about_center:
            transformer_moving.rotate_about_center(rotation_degrees=true_rotate)
        else:
            cx, cy = _cursor_pos(cursor_cds)
            transformer_moving.rotate_about_point((cy, cx), rotation_degrees=true_rotate)
        update_moving()

    scale_step_input = pn.widgets.FloatInput(name='Scale step (%):',
                                             value=1.,
                                             start=0.1,
                                             end=100.,
                                             width=125)

    def fine_scale(event, xdir=0, ydir=0):
        if not xdir and not ydir:
            return
        about_center = about_center_cbox.value
        xscale = 1 - (scale_step_input.value * xdir / 100)
        yscale = 1 - (scale_step_input.value * ydir / 100)
        if about_center:
            transformer_moving.xy_scale_about_center(xscale=xscale, yscale=yscale)
        else:
            cx, cy = _cursor_pos(cursor_cds)
            transformer_moving.xy_scale_about_point((cy, cx), xscale=xscale, yscale=yscale)
        update_moving()

    def _undo(event):
        transformer_moving.remove_transform()
        update_moving()

    undo_button = pn.widgets.Button(name='Undo',
                                    max_width=125,
                                    button_type='primary')
    undo_button.on_click(_undo)

    def getter() -> AffineTransform:
        return transformer_moving.get_combined_transform()

    return pn.Column(
        pn.Row(
            pn.Column(
                pn.Row(
                    overlay_alpha,
                    wobble_alpha_cbox,
                    wobble_step_input,
                    show_diff_cbox,
                ),
                fig_lo,
                transform_md,
            ),
            pn.Column(
                undo_button,
                translate_step_input,
                translate_buttons(fine_translate),
                about_center_cbox,
                rotate_step_input,
                rotate_buttons(fine_rotate),
                scale_step_input,
                scale_buttons(fine_scale),
                shear_step_input,
                shear_buttons(fine_shear),
            )
        )
    ), getter
