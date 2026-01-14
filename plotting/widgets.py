from matplotlib.widgets import Slider


class DoubleSlider(Slider):
    def __init__(self, ax, label, valmin, valmax, valmin_init=None, valmax_init=None, valfmt=None,
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, dragging=True, valstep=None,
                 orientation='horizontal', **kwargs):
        if valmin_init is None:
            valmin_init = valmin
        if valmax_init is None:
            valmax_init = valmax
        super().__init__(ax,  label, valmin, valmax, valinit=valmax_init, valfmt=valfmt,
                         closedmin=closedmin, closedmax=closedmax, slidermin=slidermin,
                         slidermax=slidermax, dragging=dragging, valstep=valstep,
                         orientation='horizontal', **kwargs)

        self.current_val_min = self.valmin
        self.current_val_max = self.valmax

        self.valtext.set_visible(False)
        self.vline.set_visible(False)

        self.label.set_position([0.5, 1.02])
        self.valtext_min = ax.text(-0.1, 0.5, self._format(self.current_val_min),
                                   transform=ax.transAxes,
                                   verticalalignment='center',
                                   horizontalalignment='left')

        self.valtext_max = ax.text(1.02, 0.5, self._format(self.current_val_max),
                                   transform=ax.transAxes,
                                   verticalalignment='center',
                                   horizontalalignment='left')

        self.vline_min = ax.axvline(self.current_val_min, 0, 1, color='r', lw=1)
        self.vline_max = ax.axvline(self.current_val_max, 0, 1, color='r', lw=1)

    def _update(self, event):
        """Update the slider position."""
        if self.ignore(event) or event.button != 1:
            return

        if event.name == 'button_press_event' and event.inaxes == self.ax:
            self.drag_active = True
            # event.canvas.grab_mouse(self.ax)

            if self.orientation == 'vertical':
                self.first_val = self._value_in_bounds(event.ydata)
            else:
                self.first_val = self._value_in_bounds(event.xdata)
            self.second_val = None

        if self.drag_active and (event.name != 'button_release_event'):
            if self.orientation == 'vertical':
                self.second_val = self._value_in_bounds(event.ydata)
            else:
                self.second_val = self._value_in_bounds(event.xdata)
            if self.second_val not in [None, self.val]:
                self.set_two_val(val1=min(self.first_val, self.second_val),
                                 val2=max(self.first_val, self.second_val))
            return

        if event.name == 'button_release_event':
            self.drag_active = False
            event.canvas.release_mouse(self.ax)

            if self.orientation == 'vertical':
                self.second_val = self._value_in_bounds(event.ydata)
            else:
                self.second_val = self._value_in_bounds(event.xdata)

            if self.second_val not in [None, self.val]:
                self.set_two_val(val1=min(self.first_val, self.second_val),
                                 val2=max(self.first_val, self.second_val))


            for cid, func in self.observers.items():
                func(self.current_val_min, self.current_val_max)


    def _format_two(self, val1, val2):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return self.valfmt % val1
        else:
            _, s, _ = self._fmt.format_ticks([self.valmin, val1, val2, self.valmax])
            # fmt.get_offset is actually the multiplicative factor, if any.
            return s + self._fmt.get_offset()

    def set_two_val(self, val1, val2):
        """
        Set slider value to *val1* and *val2*

        Parameters
        ----------
        val1 : float
        val2 : float
        """

        if self.orientation == 'vertical':
            self.poly.set_xy([[0, val1], [0, val2], [1, val2],
                              [1, val1]])
        else:
            self.poly.set_xy([[val1, 0], [val2, 0],
                              [val2, 1], [val1, 1]])
            self.vline_min.set_xdata(val1)
            self.vline_max.set_xdata(val2)

        self.valtext_min.set_text(self._format(val1))
        self.valtext_max.set_text(self._format(val2))
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.current_val_min = val1
        self.current_val_max = val2
        if not self.eventson:
            return
