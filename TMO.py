import typing
import numpy as np
import pyqtgraph as pg
import ami.graph_nodes as gn
from amitypes import Array1d, Array2d
from ami.flowchart.library.common import CtrlNode
from ami.flowchart.library.DisplayWidgets import Histogram2DWidget
from pyqtgraph import QtCore
from psana.pop.POP import POP as psanaPOP


class Histogram2DRoi(CtrlNode):

    """
    Roi of 2d histogram
    """

    nodeName = "Histogram2DRoi"
    uiTemplate = [('origin x', 'intSpin', {'value': 0, 'min': 0}),
                  ('origin y', 'intSpin', {'value': 0, 'min': 0}),
                  ('extent x', 'intSpin', {'value': 10, 'min': 1}),
                  ('extent y', 'intSpin', {'value': 10, 'min': 1})]

    def __init__(self, name):
        super().__init__(name, terminals={'XBins': {'io': 'in', 'ttype': Array1d},
                                          'YBins': {'io': 'in', 'ttype': Array1d},
                                          'Counts': {'io': 'in', 'ttype': Array2d},
                                          'XBins.Out': {'io': 'out', 'ttype': Array1d},
                                          'YBins.Out': {'io': 'out', 'ttype': Array1d},
                                          'Counts.Out': {'io': 'out', 'ttype': Array2d}},
                         viewable=True)

    def display(self, topics, terms, addr, win, **kwargs):
        super().display(topics, terms, addr, win, Histogram2DWidget, **kwargs)

        if self.widget:
            self.roi = pg.RectROI([self.values['origin x'], self.values['origin y']],
                                  [self.values['extent x'], self.values['extent y']])
            self.roi.sigRegionChangeFinished.connect(self.set_values)
            self.widget.view.addItem(self.roi)

        return self.widget

    def set_values(self, *args, **kwargs):
        # need to block signals to the stateGroup otherwise stateGroup.sigChanged
        # will be emmitted by setValue causing update to be called
        self.stateGroup.blockSignals(True)
        roi = args[0]
        extent, _, origin = roi.getAffineSliceParams(self.widget.imageItem.image, self.widget.imageItem)
        origin = QtCore.QPoint(*origin)*self.widget.transform
        extent = QtCore.QPoint(*extent)
        self.values['origin x'] = origin.x()
        self.values['origin y'] = origin.y()
        self.values['extent x'] = extent.x()
        self.values['extent y'] = extent.y()
        self.ctrls['origin x'].setValue(self.values['origin x'])
        self.ctrls['extent x'].setValue(self.values['extent x'])
        self.ctrls['origin y'].setValue(self.values['origin y'])
        self.ctrls['extent y'].setValue(self.values['extent y'])
        self.stateGroup.blockSignals(False)
        self.sigStateChanged.emit(self)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

        if self.widget:
            origin = QtCore.QPoint(self.values['origin x'], self.values['origin y'])
            self.roi.setPos(origin.x(), origin.y(), finish=False)
            self.roi.setSize((self.values['extent x'], self.values['extent y']), finish=False)

    def to_operation(self, inputs, conditions={}):
        outputs = self.output_vars()

        ox = self.values['origin x']
        ex = self.values['extent x']
        oy = self.values['origin y']
        ey = self.values['extent y']

        def func(x, y, img):
            xstart = np.digitize(ox, x)
            ystart = np.digitize(oy, y)
            xs = slice(xstart, xstart+ex)
            ys = slice(ystart, ystart+ey)
            return x[xs], y[ys], img[xs, ys]

        node = gn.Map(name=self.name()+"_operation",
                      condition_needs=conditions, inputs=inputs, outputs=outputs,
                      func=func,
                      parent=self.name())
        return node


class POPProc(object):

    def __init__(self, args):
        self.args = args
        # self.calibconsts = {}
        self.accum_num = args.pop('accum_num', 1)
        self.normalize_dist = args.pop('normalizeDist', True)
        self.proc = None
        self.img = None
        self.counter = 0

    def __call__(self, img):
    #     init = False
    #     if self.calibconsts.keys() != calib.keys():
    #         init = True
    #     elif not all(np.array_equal(self.calibconsts[key], calib[key]) for key in calib):
    #         init = True

        if self.proc is None:
            self.img = np.zeros(img.shape)
            self.counter += 1
            self.proc = psanaPOP(img=img, **self.args)

            self.slice_img = np.array([[np.nan]])
            self.rbins = np.array([np.nan, np.nan])
            self.distr = np.array([np.nan])
            self.ebins = np.array([np.nan, np.nan])
            self.diste = np.array([np.nan])

        if self.accum_num == 1:
            pop = self.proc
            pop.Peel(img)
            slice_img = pop.GetSlice()
            rbins, distr = pop.GetRadialDist()
            ebins, diste = pop.GetEnergyDist()
            if self.normalize_dist:
                distr = distr/distr.max()
                diste = diste/diste.max()

            return slice_img, rbins, distr, ebins, diste

        self.img += img

        if self.counter % self.accum_num == 0:
            pop = self.proc
            pop.Peel(self.img)
            self.slice_img = pop.GetSlice()
            self.rbins, self.distr = pop.GetRadialDist()
            self.ebins, self.diste = pop.GetEnergyDist()
            self.counter = 0
            self.img = np.zeros(img.shape)
            if self.normalize_dist:
                self.distr = self.distr/self.distr.max()
                self.diste = self.diste/self.diste.max()
        else:
            self.counter += 1

        return self.slice_img, self.rbins, self.distr, self.ebins, self.diste


class POP(CtrlNode):

    """
    psana POP
    """

    nodeName = "POP"

    uiTemplate = [('RBFs_fnm', 'text'),
                  ('lmax', 'intSpin', {'value': 4, 'values': ['2', '4', '6', '8', '10', '12']}),
                  ('reg', 'doubleSpin', {'value': 0}),
                  ('alpha', 'doubleSpin', {'value': 4e-4}),
                  ('X0', 'intSpin', {'value': 512}),
                  ('Y0', 'intSpin', {'value': 512}),
                  ('Rmax', 'intSpin', {'value': 512}),
                  ('edge_w', 'intSpin', {'value': 10}),
                  ('accum_num', 'intSpin', {'value': 30, 'min': 0}),
                  ('normalizeDist', 'check', {'checked': True})]

    def __init__(self, name):
        super().__init__(name, terminals={'Image': {'io': 'in', 'ttype': Array2d},
                                          'sliceImg': {'io': 'out', 'ttype': Array2d},
                                          'Rbins': {'io': 'out', 'ttype': Array1d},
                                          'DistR': {'io': 'out', 'ttype': Array1d},
                                          'Ebins': {'io': 'out', 'ttype': Array1d},
                                          'DistE': {'io': 'out', 'ttype': Array1d}})

    def to_operation(self, inputs, conditions={}):
        outputs = self.output_vars()

        node = gn.Map(name=self.name()+"_operation",
                      condition_needs=conditions,
                      inputs=inputs, outputs=outputs, parent=self.name(),
                      func=POPProc(self.values))
        return node
