#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
(experimental) Importer for HDE file format

2010-05-20 Stefan Schmidt

"""
import sys, os
import struct
from numpy import *
from pylab import figure, clf, matshow, plot, gray, legend, draw, show, title, gca, savefig




HEADERSIZE = 2048
BHEADERSIZE = 256
GAMMA = 4.0  # gamma correction factor for visualization, as recommended by HDE


class HeaderInfo:
    def __init__(self, headerstring):
        # check header signature: (version 102 or 103)
        #if headerstring[:12] != 'HSF-OCT-102\x00' and headerstring[:12] != 'HSF-OCT-103\x00':
        #    raise RuntimeError("Header unreadable")

        fields = [('12xi', "SizeX"),
                  ('i', "NumBScans"),
                  ('i', "SizeZ"),
                  ('d', "ScaleX"),
                  ('d', "Distance"),
                  ('d', "ScaleZ"),
                  ('i', "SizeXSlo"),
                  ('i', "SizeYSlo"),
                  ('d', "ScaleXSlo"),
                  ('d', "ScaleYSlo"),
                  ('i', "FieldSizeSlo"),
                  ('d', "ScanFocus"),
                  ('4s', "ScanPosition"),
                  ('Q', "ExamTime"),
                  ('i', "ScanPattern"),
                  ('i', "BScanHdrSize"),
                  ('16s', "ID"),
                  ('16s', "ReferenceID")]
        # from version 101 on:
        fields += [('i', "PID"),
                   ('21s3x', "PatientID"),
                   ('d', "DOB"),  # date of birth, a 64bit float according to MS Date spec.
                   ('i', "VID"),
                   ('24s', "VisitID"),
                   ('d', "VisitDate")]
        # from version 102 on:
        fields += [('i', "GridType"),
                   ('i1832x', "GridOffset")]

        fmt = "".join([i[0] for i in fields])
        names = [i[1] for i in fields]

        res = struct.unpack('=' + fmt, headerstring)
        self.__dict__.update(zip(names, res))


class BScanHeaderInfo:
    def __init__(self, headerstring):
        # check header signature: (version 102)
        # if headerstring[:12] != 'HSF-BS-102\x00':
        #    raise RuntimeError("B-Scan Header unreadable")
        # if headerstring[:7] != 'HSF-BS-':
        #     raise RuntimeError("B-Scan Header unreadable")
        #
        fields = [('12xi', "BScanHdrSize"),
                  ('d', "StartX"),
                  ('d', "StartY"),
                  ('d', "EndX"),
                  ('d', "EndY"),
                  ('i', "NumSeg"),
                  ('i', "OffSeg")]
        # from version 101 on:
        fields += [('f', "Quality")]
        # from version 102 on:
        fields += [('i192x', "Shift")]
        # the remainder contains segmentation data. Skip it for now
        fmt = "".join([i[0] for i in fields])
        names = [i[1] for i in fields]
        res = struct.unpack('=' + fmt, headerstring[:struct.calcsize('=' + fmt)])
        self.__dict__.update(zip(names, res))


class BScan:
    def show(self, fignum=None, titletext=None):
        if fignum is not None:
            figure(fignum)
            clf()
        u = self.data ** (1.0 / GAMMA)
        u[u > 1e5] = 0.0
        matshow(u, fignum=fignum)
        gray()
        if titletext is not None:
            title(titletext)

        # draw segments if available:
        colors = "ycmrbw"
        layernames = ["ILM", "RPE", "NFL"]
        for segnr, seg in enumerate(self.segments):
            col = colors[mod(segnr, len(colors))]
            plot(arange(self.headerinfo.SizeX), seg, "-%c" % col, scalex=False, scaley=False, hold=True)
        legend(layernames[:len(self.segments)], loc=0)


class OCTScan:
    def __init__(self, data):
        header = data[:HEADERSIZE]
        self.headerinfo = headerinfo = HeaderInfo(header)

        offset = HEADERSIZE

        # read SLO image (byte format)
        slo_size = headerinfo.SizeXSlo * headerinfo.SizeYSlo
        self.SLO = frombuffer(data, dtype='B', count=slo_size, offset=offset).reshape(
            (headerinfo.SizeXSlo, headerinfo.SizeYSlo))
        offset += slo_size

        # print "Scan pattern:",headerinfo.ScanPattern

        # read B-Scans
        self.bscans = []
        for bscannr in range(headerinfo.NumBScans):
            # read header:
            BsBlkSize = headerinfo.BScanHdrSize + headerinfo.SizeX * headerinfo.SizeZ * 4
            bheader = BScanHeaderInfo(data[offset:offset + BHEADERSIZE])
            offset += BHEADERSIZE

            # read segments if any:
            segsize = bheader.NumSeg * headerinfo.SizeX
            segments = frombuffer(data, dtype=float32, offset=offset, count=segsize).reshape(
                (bheader.NumSeg, headerinfo.SizeX))
            offset += bheader.BScanHdrSize - BHEADERSIZE

            scan = BScan()
            scan.segments = segments
            scan.headerinfo = headerinfo
            scan.bheader = bheader

            scan.data = frombuffer(data, dtype=float32, offset=offset,
                                   count=headerinfo.SizeX * headerinfo.SizeZ).reshape(
                (headerinfo.SizeZ, headerinfo.SizeX))
            offset += headerinfo.SizeX * headerinfo.SizeZ * 4

            self.bscans.append(scan)

    def show(self):
        figure(99)
        clf()
        matshow(self.SLO, fignum=99)
        gray()

        # show where the B-Scans where located in the overview image
        for nr, scan in enumerate(self.bscans):
            xdata = array((scan.bheader.StartX, scan.bheader.EndX)) / self.headerinfo.ScaleXSlo
            ydata = array((scan.bheader.StartY, scan.bheader.EndY)) / self.headerinfo.ScaleYSlo
            plot(xdata, ydata, 'x-y', scalex=False, scaley=False, hold=True)

            if self.headerinfo.ScanPattern == 2:  # circular pattern
                from matplotlib.patches import Ellipse
                radius = abs(scan.bheader.StartX - scan.bheader.EndX)
                ell = Ellipse((xdata[1], ydata[1]), 2.0 * radius / self.headerinfo.ScaleXSlo,
                              2.0 * radius / self.headerinfo.ScaleYSlo, edgecolor='m', fill=0)
                gca().add_patch(ell)
        draw()


# some plotting helper functions
# (originally in external module, now here in self-contained form)

def exportfig(basefn, formats=['png', 'pdf'], mangleFilenames=True):
    # mangle filename (latex-compatible):
    if mangleFilenames:
        import os
        basefn = os.path.sep.join(basefn.split(os.path.sep)[:-1] + [basefn.split(os.path.sep)[-1].replace(".", "_")])
    for suffix in formats:
        savefig("%s.%s" % (basefn, suffix))
    # gzip the eps version:
    # if "eps" in formats:
    #     import logging
    #     try:
    #         import os
    #         os.system("gzip -f '%s'" % (basefn + ".eps"))
    #     except Exception, e:
    #         logging.warning("Could not compress figure to eps.gz" + str(e))
    return basefn


def exportfig_raw(basefn):
    #try:
        gca().images[0].write_png(basefn + "_raw.png")
    # except Exception, e:
    #     import logging
    #     logging.warning("Could not write %s_raw.png:" % basefn + str(e))

#
if __name__ == "__main__":
    # args = ["C:\temp\data_BioTree\JW_J_1_Angio.vol"]
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options]  FILE.vol")
    parser.add_option("-m", "--matlabexport", dest="matoutput", help="Output data to matlab-readable file",
                      metavar="FILE")
    parser.add_option("-s", "--show", dest="show", action="store_true", default=False)
    parser.add_option("-v", "--showVol", dest="showVol", action="store_true", default=False)
    parser.add_option("-b", "--batch", dest="batch", action="store_true", default=False,
                      help="modifies show: exit after any export operations")
    parser.add_option("-g", "--gather", dest="gather", action="store_true", default=False,
                      help="Gather segmentation sample")
    parser.add_option("-p", "--png", dest="png", help="Write png files", metavar="FILE")
    parser.add_option("-P", "--pdf", dest="pdf", help="Write pdf files", metavar="FILE")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.error("missing options, try --help")

    scans = []
    samples = []
    imsamples = []
    for fn in args:
        print("Loading file ", fn)

        oct = OCTScan(open(fn, "rb").read())
        oct.filename = fn.split(os.path.sep)[-1]

        if options.show:
            oct.show()
            if options.png:
                exportfig_raw(options.png + "-SLO")
            if options.pdf:
                exportfig(options.pdf + "-SLO")

            # show B-scans
            for nr, scan in enumerate(oct.bscans):
                if nr in range(20,25):
                    scan.show(fignum=1 + nr, titletext="B-Scan %d" % (1 + nr))
                    draw()
                    if options.png:
                        exportfig_raw(options.png + "-B%d" % nr)
                    if options.pdf:
                        exportfig(options.pdf + "-B%d" % nr)

        if options.matoutput:
            from scipy.io import savemat

            matdata = {}
            matdata["SLO"] = oct.SLO
            matdata.update(oct.headerinfo.__dict__)
            for nr, scan in enumerate(oct.bscans):
                matdata["B%d" % nr] = scan.data
                matdata["B%dseg" % nr] = scan.segments
                matdata.update(scan.bheader.__dict__)
            matdata["TESTSCALAR"] = 42
            savemat(options.matoutput, matdata)  # ,format="5") ##do_compression=True, ,oned_as='row'

        if options.gather:
            for nr, scan in enumerate(oct.bscans):
                imsamples.append(scan.data)
                samples.append(scan.segments)
            scans.append(oct)

    if options.gather:
        samples = array(samples)

    if options.show:
        if options.gather:
            figure()
            plot(samples[:, 1, :].T)
        if not options.batch:
            show()

    if options.showVol:
        from mayavi import mlab

        v = array([b.data for b in oct.bscans])
        del oct
        v[v > 1.0] = 0.0  # handle invalid data at borders
        if 0:
            print("Filtering data:")
            # smooth and downsample/extract part:
            from scipy.ndimage import gaussian_filter

            v = gaussian_filter(v, sigma=1.5)
            # v = v[::3,::3,::3].copy()
            v = v[:v.shape[0] / 3, :v.shape[1] / 2, :v.shape[2] / 3].copy()

        v = v**(1.0/GAMMA)
        v = (v * 255.0).astype('uint8')
        # print("Creating view, size=", v.shape)
        mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))
        mlab.clf()

        source = mlab.pipeline.scalar_field(v)
        min = v.min()
        max = v.max()
        print(min)
        print(max)
        # vol = mlab.pipeline.volume(source, vmin=min + 0.65 * (max - min), vmax=min + 0.9 * (max - min))
        vol = mlab.pipeline.volume(source)

        # mlab.view(132, 54, 45, [21, 20, 21.5])
        # mlab.colorbar()
        print(mlab.view())
        mlab.view(270)

        print(mlab.view())
        mlab.show()
