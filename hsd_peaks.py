from typing import List
from amitypes import Array1d, HSDPeaks
from ami.flowchart.library.common import CtrlNode
from ami.flowchart.library.Editors import ChannelEditor
from ami.flowchart.library.DisplayWidgets import PlotWidget, symbols_colors
import ami.graph_nodes as gn
import numpy as np


class HSDPeakSelector(CtrlNode):

    """
    Select peaks from HSD
    """

    nodeName = "HSDPeakSelector"
    uiTemplate = [('digitizer', 'intSpin', {'value': 0, 'min': 0}),
                  ('channel', 'intSpin', {'value': 0, 'min': 0, 'max': 16})]

    def __init__(self, name):
        super().__init__(name, terminals={

            'In': {'io': 'in', 'ttype': HSDPeaks},
            'Start Pos': {'io': 'out', 'ttype': List[int]},
            'Times': {'io': 'out', 'ttype': List[Array1d]},
            'Peaks': {'io': 'out', 'ttype': List[Array1d]},
            'Num Peaks': {'io': 'out', 'ttype': int}
        })

    def to_operation(self, **kwargs):
        digitizer = self.values['digitizer']
        channel = self.values['channel']

        def func(d):
            peaks = d[digitizer][channel]
            start_pos = peaks[0]
            peaks = peaks[1]
            times = []
            for start, peak in zip(start_pos, peaks):
                times.append(np.arange(start, start+len(peak)))

            return start_pos, times, peaks, len(peaks)

        node = gn.Map(name=self.name()+"_operation",
                      **kwargs, func=func)

        return node


try:
    import psana.hexanode.WFPeaks as psWFPeaks

    class HSDWFPeaks(CtrlNode):

        """
        WFPeaks
        """

        nodeName = "HSDWFPeaks"

        def __init__(self, name):
            super().__init__(name, terminals={'Times': {'io': 'in', 'ttype': List[Array1d]},
                                              'Waveform': {'io': 'in', 'ttype': List[Array1d]},
                                              'Num of Hits': {'io': 'out', 'ttype': Array1d},
                                              'Index': {'io': 'out', 'ttype': Array1d},
                                              'Values': {'io': 'out', 'ttype': Array1d},
                                              'Peak Times': {'io': 'out', 'ttype': Array1d}})
            self.values = {}

        def display(self, topics, terms, addr, win, **kwargs):
            if self.widget is None:
                self.widget = ChannelEditor(parent=win)
                self.values = self.widget.values
                self.widget.sigStateChanged.connect(self.state_changed)

            return self.widget

        def to_operation(self, **kwargs):
            numchs = len(self.widget.channel_groups)
            cfdpars = {'numchs': numchs,
                       'numhits': self.values['num hits'],
                       'DLD': self.values['DLD'],
                       'version': 4,
                       'cfd_wfbinbeg': self.values['cfd_wfbinbeg'],
                       'cfd_wfbinend': self.values['cfd_wfbinend']}

            paramsCFD = {}
            for chn in range(0, numchs):
                paramsCFD[chn] = self.values[f"Channel {chn}"]

            cfdpars['paramsCFD'] = paramsCFD
            wfpeaks = psWFPeaks.WFPeaks(**cfdpars)

            def peakFinder(wts, wfs):
                hits = []
                index = []
                values = []
                peak_times = []

                for time, waveform in zip(wts, wfs):
                    time = time.reshape(1, -1)
                    waveform = waveform.reshape(1, -1)
                    phits, pindex, pvalues, ppeak_times = wfpeaks(waveform, time)
                    if phits:
                        hits.append(phits[0])
                        index.append(pindex.reshape(-1))
                        values.append(pvalues.reshape(-1))
                        peak_times.append(ppeak_times.reshape(-1))

                if hits:
                    return np.array(hits), np.concatenate(index), np.concatenate(values), np.concatenate(peak_times)
                else:
                    return np.array([]), np.array([]), np.array([]), np.array([])

            node = gn.Map(name=self.name()+"_operation", **kwargs, func=peakFinder)
            return node

except ImportError as e:
    print(e)


class PeakWidget(PlotWidget):

    def __init__(self, topics=None, terms=None, addr=None, parent=None, **kwargs):
        super().__init__(topics, terms, addr, parent=parent, **kwargs)

    def data_updated(self, data):
        i = 0
        times = self.terms["Times"]
        peaks = self.terms["Peaks"]

        name = " vs ".join((peaks, times))

        times = np.concatenate(data[times])
        peaks = np.concatenate(data[peaks])

        if name not in self.plot:
            symbol, color = symbols_colors[i]
            idx = f"trace.{i}"
            i += 1
            legend_name = self.update_legend_layout(idx, name, symbol=symbol, color=color, style='None')
            attrs = self.legend_editors[idx].attrs
            self.trace_attrs[name] = attrs
            self.plot[name] = self.plot_view.plot(x=times, y=peaks,
                                                  name=legend_name, pen=attrs['pen'], **attrs['point'])
        else:
            attrs = self.trace_attrs[name]
            self.plot[name].setData(x=times, y=peaks, **attrs['point'])


class PeakViewer(CtrlNode):

    """
    WaveformViewer displays 1D arrays.
    """

    nodeName = "PeakViewer"
    uiTemplate = []

    def __init__(self, name):
        super().__init__(name, terminals={"Times": {"io": "in", "ttype": List[Array1d]},
                                          "Peaks": {"io": "in", "ttype": List[Array1d]}},
                         viewable=True)

    def display(self, topics, terms, addr, win, **kwargs):
        return super().display(topics, terms, addr, win, PeakWidget, **kwargs)
