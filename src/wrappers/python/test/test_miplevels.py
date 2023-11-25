import unittest

import PyOpenEXR
import numpy as np

class testReadWrite(unittest.TestCase):

    def setUp(self):
        in_path = './test/ColorCodedLevels.exr'
        self.in_file = PyOpenEXR.MultiPartInputFile(in_path)

    def testReadMipWriteLine(self):

        out_path_t = './test/ColorCodedLevels_separated_{:04d}_l.exr'
        for part_index in range(self.in_file.parts()):
            in_header = self.in_file.header(part_index)
            #print(h.name())
            #print(h.type())
            #print(h.getCompression())

            input_part = PyOpenEXR.TiledInputPart(self.in_file)
            for level_index in range(input_part.numLevels()):
                pixels = input_part.readPixels(level_index)
                w = input_part.levelWidth(level_index)
                h = input_part.levelHeight(level_index)
                out_header = PyOpenEXR.Header(w, h, 1.0)
                out_header.setType(PyOpenEXR.SCANLINEIMAGE)
                out_header.setCompression(PyOpenEXR.Compression.ZIP)
                channel_names = in_header.getChannels()
                
                #out_header.setChannels(PyOpenEXR.PixelType.HALF, channel_names)
                out_header.setChannels(in_header.pixelType(), channel_names)
                out_file = PyOpenEXR.MultiPartOutputFile(out_path_t.format(h), [out_header])
                out_part = PyOpenEXR.OutputPart(out_file)
                out_part.writePixels(pixels)


    def testReadMipWriteTile(self):

        out_path_t = './test/ColorCodedLevels_separated_{:04d}_t.exr'
        for part_index in range(self.in_file.parts()):
            in_header = self.in_file.header(part_index)
            if in_header.hasTileDescription():
                desc = in_header.tileDescription()
                #print(desc.x_size, desc.y_size, desc.mode, desc.rounding_mode)

            input_part = PyOpenEXR.TiledInputPart(self.in_file)

            for level_index in range(input_part.numLevels()):
                pixels = input_part.readPixels(level_index)
                w = input_part.levelWidth(level_index)
                h = input_part.levelHeight(level_index)
                out_header = PyOpenEXR.Header(w, h, 1.0)
                flat_tile_desc = PyOpenEXR.TileDescription(8, 8, PyOpenEXR.LevelMode.ONE_LEVEL)
                out_header.setType(PyOpenEXR.TILEDIMAGE)
                out_header.setTileDescription(flat_tile_desc)
                out_header.setCompression(PyOpenEXR.Compression.ZIP)
                channel_names = in_header.getChannels()
                out_header.setChannels(in_header.pixelType(), channel_names)
                out_file = PyOpenEXR.MultiPartOutputFile(out_path_t.format(h), [out_header])
                out_part = PyOpenEXR.TiledOutputPart(out_file)
                out_part.writePixels(pixels)


if __name__ == "__main__":
    unittest.main()
