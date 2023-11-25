import sys
import unittest

from pprint import pprint
import PyOpenEXR
from PyOpenEXR import Vec2i, Vec2f, Box2i, Box2f
import numpy as np

class testReadWrite(unittest.TestCase):

    def testHeader(self):
        header = PyOpenEXR.Header(64, 64, 1.5)
        self.assertEqual(header.pixelAspectRatio(), 1.5)
        #self.assertEqual(header.displayWindow(), Box2i(Vec2i(0, 0), Vec2i(63, 63)))
        print(header.displayWindow())

    #def testAttributes(self):
    #    pass

    def testTypes(self):
        v2i = Vec2i(64, 64)
        v2f = Vec2f(1.0, 2.0)
        box2i = Box2i(Vec2i(-2, 2), Vec2i(-4, 4))
        box2f = Box2f(Vec2f(-0.2, 0.2), Vec2f(-0.4, 0.4))
        self.assertEqual(v2i.x, 64)
        self.assertEqual(v2i.y, 64)
        self.assertEqual(box2i.min.x, -2)
        self.assertEqual(box2i.min.y, 2)
        self.assertEqual(box2i.max.x, -4)
        self.assertEqual(box2i.max.y, 4)
        self.assertAlmostEqual(v2f.x, 1.0)
        self.assertAlmostEqual(v2f.y, 2.0)
        self.assertAlmostEqual(box2f.min.x, -0.2)
        self.assertAlmostEqual(box2f.min.y, 0.2)
        self.assertAlmostEqual(box2f.max.x, -0.4)
        self.assertAlmostEqual(box2f.max.y, 0.4)

    def testMain(self):
        #in_path = 'C:/Users/Resartist/Documents/maya/projects/default/images/temp/persp_thing_scan.exr'
        in_path = 'C:/Users/Resartist/Documents/maya/projects/default/images/temp/persp_thing_tiled.exr'
        out_path = 'C:/Users/Resartist/Documents/maya/projects/default/images/temp/persp_thing_tiled_out.exr'

        in_file = PyOpenEXR.MultiPartInputFile(in_path)
        headers = [in_file.header(i) for i in range(in_file.parts())]

        headers = []
        for i in range(in_file.parts()):
            header = in_file.header(i)
            header.setCompression(PyOpenEXR.Compression.PIZ)
            header.erase('arnold/stats/geo/triangles')
            #print(header.name())
            #print(header.type())
            #print(header.getCompression())
            headers.append(header)


        identity = np.identity(4, dtype=np.float32)
        # convert our numpy array to a PyOpenEXR.Matrix44f by flattening it.
        retranslate = PyOpenEXR.Matrix44f(*identity.ravel())
        headers[0].setAttribute('first_identity', retranslate)
        [h.setAttribute('shared_float', 8.10) for h in headers]

        out_file = PyOpenEXR.MultiPartOutputFile(out_path, headers)

        if headers[0].hasTileDescription():
            i_part_type = PyOpenEXR.TiledInputPart
            o_part_type = PyOpenEXR.TiledOutputPart
        else:
            i_part_type = PyOpenEXR.InputPart
            o_part_type = PyOpenEXR.OutputPart

        for i in range(in_file.parts()):
            header.setCompression(PyOpenEXR.Compression.PIZ)
            input_part = i_part_type(in_file, i)
            header = input_part.header()
            dtw = header.dataWindow()
            dpw = header.displayWindow()

            #print(header.hasName())
            #print(header.name())
            #header.hasType()
            #print(header.type())
            channel_names = header.getChannels()

            attributes = header.getAttributes()
            #pprint(attributes)
            mat4 = attributes.get('worldToCamera')
            if mat4:
                array = mat4.asNumpy()

            #PyOpenEXR.Matrix44(array.flatten())
            #fps = attributes.get('framesPerSecond')
            #timcode = attributes.get('timeCode')
            #par = PyOpenEXR.getAttribute(header, 'framesPerSecond')
            #if timcode:
            #    print('timecode hours:', timcode.hours())
            #    tc = PyOpenEXR.TimeCode(1, 2, 3, 4)
            #    print('timecode hours:', tc.hours())
            '''
            if fps:
                print(fps.n, fps.d)
                print(float(fps))
            else:
                #fps = PyOpenEXR.Rational(24000, 1001)
                fps = PyOpenEXR.Rational(23.976023976023978)
                print(float(fps))
                print(fps.n, fps.d)
            '''
            pixels = input_part.readPixels(0)
            print(pixels.shape, pixels.dtype, header.name(), channel_names)

            #print(pixels[141][682])
            #print(pixels[540-141-1][682])
            #if header.type() == PyOpenEXR.InputPartType.TILEDIMAGE:
            #if header.hasTileDescription():
            #if header.type() == "tiledimage":

            out_part = o_part_type(out_file, i)

            if pixels.dtype == np.uint32:
                pixels -= 100000
            else:
                pixels[:, :, :3] += 0.1

            out_part.writePixels(pixels)
            #out_part.copyPixels(input_part)
            #out_part.recompress(input_part, out_part)
            #print(pixels.dtype, pixels.shape)


if __name__ == "__main__":
    unittest.main()
