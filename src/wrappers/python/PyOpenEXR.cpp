#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define IMATH_HALF_NO_LOOKUP_TABLE

// types
#include <half.h>
#include "ImathMatrix.h"
#include "ImathVec.h"
#include "ImathBox.h"

// Attribute types
#include <ImfFloatAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfIntAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfTimeCodeAttribute.h>
#include <ImfRationalAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfBoxAttribute.h>
#include <ImfVecAttribute.h>
#include <ImfCompressionAttribute.h>

#include <ImfRgbaFile.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfTileDescription.h>
#include <ImfTiledOutputFile.h>
#include <ImfTiledInputFile.h>
#include <ImfTiledInputPart.h>
#include <ImfTiledOutputPart.h>

#include <ImfArray.h>
#include <ImfTimeCode.h>
#include <ImfRational.h>
#include <ImfMultiPartInputFile.h>
#include <ImfMultiPartOutputFile.h>
#include <ImfInputPart.h>
#include <ImfOutputPart.h>
#include <ImfFrameBuffer.h>
#include <ImfChannelList.h>
#include <ImfPartType.h>

#include <string>
#include <iostream>
#include <set>
#include <functional>
#include <unordered_map>
#include <cmath>
#include <format>

namespace py = pybind11;

// This tells pybind11 how to interpret the half type.
namespace pybind11 {
namespace detail {
    // ref: https://github.com/pybind/pybind11/issues/1776#issuecomment-492742167

    constexpr int NPY_FLOAT16 = 23;

    template<> struct npy_format_descriptor<half> {
        static py::dtype dtype() {
            handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
            return reinterpret_borrow<py::dtype>(ptr);
        }
        static std::string format() {
            // following: https://docs.python.org/3/library/struct.html#format-characters
            return "e";
        }
        static constexpr auto name = _("float16");
    };

}
}  // namespace pybind11::detail

template<typename T>
inline py::array_t<T>
asNDArray(T* data, size_t width, size_t height, size_t channels) {
    py::capsule capsule(data, [](void* f) {
        // delete the pixel buffer when the numpy array goes out of scope.
        delete[](reinterpret_cast<Imf::Rgba*>(f));
    });
    // Assign same shape and strides as the Imath pixels array for passing to numpy array.
    const size_t sz = sizeof(T);
    std::vector<size_t> shape, strides;
    shape.assign({height, width, channels});
    strides.assign({
        width * channels * sz,
        channels * sz,
        sz
    });
    return py::array_t<T>(shape, strides, data, capsule);
}

template<typename T>
inline py::array_t<T>
matrixArray44(const Imath::Matrix44<T> mat44) {
    py::array_t<float> matrix({4, 4});

    // assign the matrix data to our new array.
    for (int row = 0; row < 4; row++) {
        for (int column = 0; column < 4; ++column) {
            *matrix.mutable_data(row, column) = mat44[row][column];
        }
    }
    return matrix;
}

template<typename T>
inline py::array_t<T>
matrixArray33(const Imath::Matrix33<T> mat33) {
    py::array_t<float> matrix({3, 3});

    // assign the matrix data to our new array.
    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; ++column) {
            *matrix.mutable_data(row, column) = mat33[row][column];
        }
    }
    return matrix;
}

using SizeMap = std::unordered_map<Imf::PixelType, size_t>;
SizeMap SIZE_MAP = {
    {Imf::HALF, 2},
    {Imf::FLOAT, 4},
    {Imf::UINT, 4},
};


template <typename T>
py::object
constructNDArray(
    const T buffer,
    const size_t width,
    const size_t height,
    const uint32_t channels,
    const Imf::PixelType pixel_type
    ){

    switch (pixel_type) {
    case Imf::HALF: {
        half* data = reinterpret_cast<half*>(buffer);
        return asNDArray(data, width, height, channels);
    }
    case Imf::FLOAT: {
        float* data = reinterpret_cast<float*>(buffer);
        return asNDArray(data, width, height, channels);
    }
    case Imf::UINT: {
        uint32_t* data = reinterpret_cast<uint32_t*>(buffer);
        return asNDArray(data, width, height, channels);
    }
    default:
        throw std::runtime_error("Unsupported pixel type.");
    }
}


py::object
readPart(Imf::InputPart& part) {
    py::gil_scoped_release release; // Release Python's GIL (Global Interpreter Lock)

    Imf::Header header = part.header();
    Imath::Box2i extent = header.dataWindow();
    Imath::Vec2<int> shape = extent.size();
    shape += Imath::Vec2<int>(1, 1);

    // Prepare channel information.
    uint32_t channel_count = 0;
    std::vector<std::string> channel_names;
    Imf::PixelType pixel_type;
    auto channels = header.channels();
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        pixel_type = it.channel().type;
        channel_names.push_back(it.name());
        channel_count++;
    }

    const size_t
        bits = SIZE_MAP.at(pixel_type),
        length = shape.x * shape.y * channel_count * bits,
        x_stride = channel_count * bits,
        y_stride = shape.x * x_stride
        ;

    // allocate memory for pixels.
    char* pixel_buffer = new char[length];

    // set the frame buffer slices to read per-channel components into the pixel_buffer.

    Imf::FrameBuffer framebuffer;
    size_t offset = 0;
    for (int i = channel_count - 1; i >= 0; i--) { // reverse order
        Imf::Slice slice(pixel_type, &pixel_buffer[offset], x_stride, y_stride);
        framebuffer.insert(channel_names[i], slice);
        offset += bits;
    }

    part.setFrameBuffer(framebuffer);
    part.readPixels(extent.min.y, extent.max.y);

    py::gil_scoped_acquire acquire;

    return constructNDArray(pixel_buffer, shape.x, shape.y, channel_count, pixel_type);
};


py::object
readPartTiled(Imf::TiledInputPart& part, const int level_index = 0) {
    py::gil_scoped_release release; // Release Python's GIL (Global Interpreter Lock)

    Imf::Header header = part.header();
    Imath::Box2i extent = header.dataWindow();
    Imath::Vec2<int> shape = extent.size();
    shape += Imath::Vec2<int>(1, 1);

    // Prepare channel information.
    uint32_t channel_count = 0;
    std::vector<std::string> channel_names;
    Imf::PixelType pixel_type;
    auto channels = header.channels();
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        pixel_type = it.channel().type;
        channel_names.push_back(it.name());
        channel_count++;
    }

    const int mip_level = pow(2, level_index);
    const size_t width = shape.x / mip_level;
    const size_t height = shape.y / mip_level;

    const size_t
        bits = SIZE_MAP.at(pixel_type),
        length = width * height * channel_count * bits,
        x_stride = channel_count * bits,
        y_stride = width * x_stride
        ;

    // allocate memory for pixels.
    char* buffer = new char[length];

    // set the frame buffer slices to read per-channel components into.
    Imf::FrameBuffer framebuffer;
    size_t offset = 0;
    for (int i = channel_count - 1; i >= 0; i--) { // reverse order
        Imf::Slice slice(pixel_type, buffer + offset, x_stride, y_stride);
        framebuffer.insert(channel_names[i], slice);
        offset += bits;
    }

    part.setFrameBuffer(framebuffer);

    const uint32_t x_tile = (part.numXTiles() - 1) / mip_level;
    const uint32_t y_tile = (part.numYTiles() - 1) / mip_level;

    part.readTiles(0, x_tile, 0, y_tile, level_index);
    py::gil_scoped_acquire acquire; // Acquire Python's GIL (Global Interpreter Lock).

    return constructNDArray(buffer, width, height, channel_count, pixel_type);
};

py::list
getChannels(const Imf::Header header) {
    // open the file and calculate the dispaly window.
    auto channels = header.channels();
    py::list channel_list;
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        channel_list.insert(0, it.name());
    }
    return channel_list;
};

py::object
getAttribute(const Imf::Header header, const std::string name) {
    auto attr = header.findTypedAttribute<Imf::FloatAttribute>(name);
    //auto attr = header.findTypedAttribute<Imf::ImfRationalAttribute>(name);
    py::list result;
    if (attr)
        result.append(attr->value());

    return result;
};

template<typename T>
void toValue(py::dict result, const std::string name, const Imf::Attribute& attr) {
    const T* attribute = dynamic_cast<const T*>(&attr);
    auto value = attribute->value();
    result[name.c_str()] = value;
};

using TypeMap = std::unordered_map<
    std::string, std::function<void(py::dict, const std::string, const Imf::Attribute&)>
>;

TypeMap ATTR_MAP = {
    {"string", toValue<Imf::StringAttribute>},
    {"int", toValue<Imf::IntAttribute>},
    {"float", toValue<Imf::FloatAttribute>},
    {"double", toValue<Imf::DoubleAttribute>},
    {"box2f", toValue<Imf::Box2fAttribute>},
    {"box2i", toValue<Imf::Box2iAttribute>},
    {"v3i", toValue<Imf::V3iAttribute>},
    {"v3f", toValue<Imf::V3fAttribute>},
    {"v3d", toValue<Imf::V3dAttribute>},
    {"v2i", toValue<Imf::V2iAttribute>},
    {"v2f", toValue<Imf::V2fAttribute>},
    {"v2d", toValue<Imf::V2dAttribute>},
    {"m33f", toValue<Imf::M33fAttribute>},
    {"m33d", toValue<Imf::M33dAttribute>},
    {"m44f", toValue<Imf::M44fAttribute>},
    {"m44d", toValue<Imf::M44dAttribute>},
    {"timecode", toValue<Imf::TimeCodeAttribute>},
    {"rational", toValue<Imf::RationalAttribute>},
    {"compression", toValue<Imf::CompressionAttribute>},
    // TODO: add support for the following types.
    //hdr.insert ("a15", StringVectorAttribute (a15)); // "stringvector"
    //hdr.insert ("a22", FloatVectorAttribute (a22)); // "floatvector"
    //hdr.insert ("a13", ChromaticitiesAttribute (a13));
    //hdr.insert ("a14", EnvmapAttribute (a14));
    //hdr.insert ("a24", TestOpaqueAttribute (a24));
};

py::dict
getAttributes(Imf::Header& header) {
    py::dict result;
    for (Imf::Header::ConstIterator i = header.begin(); i != header.end(); ++i) {
        const Imf::Attribute& attr = i.attribute();
        const std::string name = i.name();
        const std::string type_name = attr.typeName();
        try {
            auto converter = ATTR_MAP.at(type_name);
            converter(result, name, attr);
        }
        catch (const std::exception& e) {
            continue;
            std::cerr << "Unhandled exception: " << e.what() << std::endl;
        }
    }
    return result;
};


void recompress(Imf::InputPart& in_part, Imf::OutputPart& out_part){
    py::gil_scoped_release release;
    // open the file and calculate the dispaly window.
    Imf::Header header = in_part.header();
    Imath::Box2i dw = header.dataWindow();
    const size_t width  = dw.max.x - dw.min.x + 1;
    const size_t height = dw.max.y - dw.min.y + 1;

    auto channels = header.channels();

    uint32_t channel_count = 0;
    std::vector<std::string> channel_names;
    std::vector<Imf::PixelType> channel_types;
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        Imf::PixelType pixel_type = it.channel().type;
        channel_types.push_back(pixel_type);
        channel_names.push_back(it.name());
        channel_count++;
    }

    // Assume all channels are the same type and size.
    Imf::PixelType pixel_type = channel_types[0];
    const size_t bit_size = SIZE_MAP.at(pixel_type);

    // Allocate memory for pixels in their entirety.
    char* buffer = new char[width * height * channel_count * bit_size];

    size_t
        x_stride = bit_size * 1,
        y_stride = bit_size * width,
        offset = width * height * bit_size;

    // set the frame buffer slices to read per-channel components into.
    Imf::FrameBuffer framebuffer;
    for (int i = 0; i < channel_count; i++) {
        Imf::Slice slice(channel_types[i], &buffer[offset*i], x_stride, y_stride);
        framebuffer.insert(channel_names[i], slice);
    }

    in_part.setFrameBuffer(framebuffer);
    in_part.readPixels(dw.min.y, dw.max.y);

    // Write the pixels to the output part.
    out_part.setFrameBuffer(framebuffer);
    out_part.writePixels(height);
    py::gil_scoped_acquire acquire;
};


template<typename T>
void writePart(Imf::OutputPart& out_part, const py::array_t<T>& array){
    // Assume all channels are the same type and size.
    py::buffer_info info = array.request();

    double* data_ptr = static_cast<double*>(info.ptr);
    char* buffer = reinterpret_cast<char*>(data_ptr);

    // Header contains all the information we need.
    Imf::Header header = out_part.header();
    Imath::Box2i dw = header.dataWindow();
    const size_t
        width  = dw.max.x - dw.min.x + 1,
        height = dw.max.y - dw.min.y + 1,
        size = info.strides[2],//sizeof(T),
        x_stride = info.strides[1],
        y_stride = info.strides[0]
        //x_stride = channel_count * size,
        //y_stride = width * x_stride,
        ;
    int index = (info.shape[2] - 1) * size;
    py::gil_scoped_release release;


    // Set the frame buffer slices for per-channel pixel components.
    // NB: the channel order is reversed.
    auto channels = header.channels();
    Imf::FrameBuffer framebuffer;
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        Imf::PixelType pixel_type = it.channel().type;
        Imf::Slice slice(pixel_type, &buffer[index], x_stride, y_stride);
        framebuffer.insert(it.name(), slice);
        index -= size;
    }

    // Write the pixels to the output part.
    out_part.setFrameBuffer(framebuffer);
    out_part.writePixels(height);
    py::gil_scoped_acquire acquire;
};

template<typename T>
void writePartTiled(Imf::TiledOutputPart& part, const py::array_t<T>& array){
    py::buffer_info info = array.request();

    double* data_ptr = static_cast<double*>(info.ptr);
    char* pixel_buffer = reinterpret_cast<char*>(data_ptr);

    // Header contains all the information we need.
    Imf::Header header = part.header();
    Imath::Box2i extent = header.dataWindow();
    Imath::Vec2<int> shape = extent.size();
    shape += Imath::Vec2<int>(1, 1);

    const size_t
        width  = shape.x,
        height = shape.y,
        bits = info.strides[2],
        x_stride = info.strides[1], // channel_count * bits,
        y_stride = info.strides[0] // width * x_stride,
        ;
    int index = (info.shape[2] - 1) * bits;
    py::gil_scoped_release release;

    // Set the frame buffer slices for per-channel pixel components.
    // NB: the channel order is reversed.
    auto channels = header.channels();
    Imf::FrameBuffer framebuffer;
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        Imf::PixelType pixel_type = it.channel().type;
        Imf::Slice slice(pixel_type, &pixel_buffer[index], x_stride, y_stride);
        framebuffer.insert(it.name(), slice);
        index -= bits;
    }

    const int x_tile = part.numXTiles() - 1;
    const int y_tile = part.numYTiles() - 1;

    // Write the pixels to the output part.
    part.setFrameBuffer(framebuffer);
    part.writeTiles(0, x_tile, 0, y_tile);
    py::gil_scoped_acquire acquire;
};


void recompressTiled(Imf::TiledInputPart& in_part, Imf::TiledOutputPart& out_part){
    py::gil_scoped_release release;
    Imf::Header header = in_part.header();
    Imath::Box2i dw = header.dataWindow();
    const size_t width  = dw.max.x - dw.min.x + 1;
    const size_t height = dw.max.y - dw.min.y + 1;

    auto channels = header.channels();

    uint32_t channel_count = 0;
    std::vector<std::string> channel_names;
    std::vector<Imf::PixelType> channel_types;
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        Imf::PixelType pixel_type = it.channel().type;
        channel_types.push_back(pixel_type);
        channel_names.push_back(it.name());
        channel_count++;
    }

    // Assume all channels are the same type and size.
    Imf::PixelType pixel_type = channel_types[0];
    const size_t bit_size = SIZE_MAP.at(pixel_type);

    // Allocate memory for pixels in their entirety.
    char* buffer = new char[width * height * channel_count * bit_size];

    // Tiled image type stride.
    size_t
        x_stride = channel_count * bit_size,
        y_stride = width * x_stride,
        offset = 0;

    // Set the frame buffer slices to read per-channel components into.
    Imf::FrameBuffer framebuffer;
    for (int i = 0; i < channel_count; i++) {
        Imf::Slice slice(channel_types[i], buffer + offset, x_stride, y_stride);
        framebuffer.insert(channel_names[i], slice);
        offset += bit_size;
    }

    const int x_tile = in_part.numXTiles() - 1;
    const int y_tile = in_part.numYTiles() - 1;

    // Read the pixels from the input part.
    in_part.setFrameBuffer(framebuffer);
    in_part.readTiles(0, x_tile, 0, y_tile);

    // Write the pixels to the output part.
    out_part.setFrameBuffer(framebuffer);
    out_part.writeTiles(0, x_tile, 0, y_tile);
    py::gil_scoped_acquire acquire;
}


template <typename T>
void declare_vec2(py::module& m, std::string typestr) {
    using Class = Imath::Vec2<T>;
    std::string pyclass_name = std::string("Vec2") + typestr;
    py::class_<Class>(m, pyclass_name.c_str()) // py::buffer_protocol(), py::dynamic_attr()
        .def(py::init<>())
        .def(py::init<T>())
        .def(py::init<T, T>())
        .def_readwrite("x", &Class::x)
        .def_readwrite("y", &Class::y)
        ;
}

template <typename T>
void declare_vec3(py::module& m, std::string typestr) {
    using Class = Imath::Vec3<T>;
    std::string pyclass_name = std::string("Vec3") + typestr;
    py::class_<Class>(m, pyclass_name.c_str()) // py::buffer_protocol(), py::dynamic_attr()
        .def(py::init<>())
        .def(py::init<T>())
        .def(py::init<T, T, T>())
        .def_readwrite("x", &Class::x)
        .def_readwrite("y", &Class::y)
        .def_readwrite("z", &Class::z)
        ;
}

template <typename T>
void declare_vec4(py::module& m, std::string typestr) {
    using Class = Imath::Vec4<T>;
    std::string pyclass_name = std::string("Vec4") + typestr;
    py::class_<Class>(m, pyclass_name.c_str()) // py::buffer_protocol(), py::dynamic_attr()
        .def(py::init<>())
        .def(py::init<T>())
        .def(py::init<T, T, T, T>())
        .def_readwrite("x", &Class::x)
        .def_readwrite("y", &Class::y)
        .def_readwrite("z", &Class::z)
        .def_readwrite("w", &Class::w)
        ;
}

template <typename T>
void declare_mat3(py::module& m, std::string typestr) {
    using Class = Imath::Matrix33<T>;
    std::string pyclass_name = std::string("Matrix33") + typestr;
    py::class_<Class>(m, pyclass_name.c_str())
        .def(py::init<>())
        .def(py::init<T, T, T,
                      T, T, T,
                      T, T, T>())
        .def("asNumpy", [](const Class& mat33) {
            return matrixArray33<T>(mat33);
        })
        ;
}

template <typename T>
void declare_mat4(py::module& m, std::string typestr) {
    using Class = Imath::Matrix44<T>;
    std::string pyclass_name = std::string("Matrix44") + typestr;
    py::class_<Class>(m, pyclass_name.c_str())
        .def(py::init<>())
        .def(py::init<T, T, T, T,
                      T, T, T, T,
                      T, T, T, T,
                      T, T, T, T>())
        .def("asNumpy", [](const Class& mat44) {
            return matrixArray44<T>(mat44);
        })
        ;
}


PYBIND11_MODULE(PyOpenEXR, m) {
    m.doc() = "OpenEXR python extension";
    m.attr("TILEDIMAGE") = py::cast(Imf::TILEDIMAGE);
    m.attr("SCANLINEIMAGE") = py::cast(Imf::SCANLINEIMAGE);
    m.attr("__version__") = py::cast(OPENEXR_VERSION_STRING);
    // supported types are
    // "s", "i", "f", "d", "i64",

    declare_vec2<int>(m, "i");
    declare_vec2<float>(m, "f");
    declare_vec2<double>(m, "d");

    declare_vec3<int>(m, "i");
    declare_vec3<float>(m, "f");
    declare_vec3<double>(m, "d");

    declare_vec4<int>(m, "i");
    declare_vec4<float>(m, "f");
    declare_vec4<double>(m, "d");

    declare_mat3<float>(m, "f");
    declare_mat3<double>(m, "d");

    declare_mat4<float>(m, "f");
    declare_mat4<double>(m, "d");

    py::class_<Imath::Box2i>(m, "Box2i")
        .def(py::init<Imath::V2i, Imath::V2i>(), py::arg("min"), py::arg("max"))
        .def_readwrite("min", &Imath::Box2i::min)
        .def_readwrite("max", &Imath::Box2i::max)
        ;

    py::class_<Imath::Box2f>(m, "Box2f")
        .def(py::init<Imath::V2f, Imath::V2f>(), py::arg("min"), py::arg("max"))
        .def_readwrite("min", &Imath::Box2f::min)
        .def_readwrite("max", &Imath::Box2f::max)
        ;

    py::enum_<Imf::PixelType>(m, "PixelType")
        .value("HALF", Imf::HALF)
        .value("FLOAT", Imf::FLOAT)
        .value("UINT", Imf::UINT)
        .export_values();

    py::enum_<Imf::LevelMode>(m, "LevelMode")
        .value("ONE_LEVEL", Imf::ONE_LEVEL)
        .value("MIPMAP_LEVELS", Imf::MIPMAP_LEVELS)
        .value("RIPMAP_LEVELS", Imf::RIPMAP_LEVELS)
        .export_values();

    py::enum_<Imf::LevelRoundingMode>(m, "LevelRoundingMode")
        .value("ROUND_DOWN", Imf::ROUND_DOWN)
        .value("ROUND_UP", Imf::ROUND_UP)
        .export_values();

    py::enum_<Imf::Compression>(m, "Compression")
        .value("NONE", Imf::NO_COMPRESSION)
        .value("RLE", Imf::RLE_COMPRESSION)
        .value("ZIPS", Imf::ZIPS_COMPRESSION)
        .value("ZIP", Imf::ZIP_COMPRESSION)
        .value("PIZ", Imf::PIZ_COMPRESSION)
        .value("PXR24", Imf::PXR24_COMPRESSION)
        .value("B44", Imf::B44_COMPRESSION)
        .value("B44A", Imf::B44A_COMPRESSION)
        .value("DWAA", Imf::DWAA_COMPRESSION)
        .value("DWAB", Imf::DWAB_COMPRESSION)
        .export_values();

    py::class_<Imf::TileDescription>(m, "TileDescription")
        .def(py::init<>())
        .def(py::init<int, int, Imf::LevelMode>(),
            py::arg("x_size"), py::arg("y_size"), py::arg("mode")
        )
        .def(py::init<int, int, Imf::LevelMode, Imf::LevelRoundingMode>(),
            py::arg("x_size"), py::arg("y_size"), py::arg("mode"), py::arg("rounding_mode")
        )
        .def_readwrite("x_size", &Imf::TileDescription::xSize)
        .def_readwrite("y_size", &Imf::TileDescription::ySize)
        .def_readwrite("mode", &Imf::TileDescription::mode)
        .def_readwrite("rounding_mode", &Imf::TileDescription::roundingMode)
        ;

    py::class_<Imf::Header>(m, "Header")
        .def(py::init<int, int, float>(), py::arg("width"), py::arg("height"), py::arg("pixel_aspect_ratio"))
        .def(py::init<int, int, Imath::Box2i, float>(), py::arg("width"), py::arg("height"), py::arg("data_window"), py::arg("pixel_aspect_ratio"))
        .def(py::init<Imath::Box2i, Imath::Box2i, float>(), py::arg("display_window"), py::arg("data_window"), py::arg("pixel_aspect_ratio"))
        .def("tileDescription", [](Imf::Header& h) {return h.tileDescription();})
        .def("hasTileDescription", &Imf::Header::hasTileDescription)
        .def("setTileDescription", [](Imf::Header& h, const Imf::TileDescription& td)
            { h.setTileDescription(td); }
        )
        .def("erase", [](Imf::Header& h, const char* attr_name)
            { h.erase(attr_name); }
        )
        //.def("erase", &Imf::Header::erase)
        .def("pixelAspectRatio", [](Imf::Header& h) {return h.pixelAspectRatio();})
        .def("displayWindow", [](Imf::Header& h) {return h.displayWindow();})
        .def("dataWindow", [](Imf::Header& h) {return h.dataWindow();})
        .def("channels", [](Imf::Header& h) {return h.channels();})
        .def("name", [](Imf::Header& h) {return h.name();})
        .def("hasName", [](Imf::Header& h) {return h.hasName();})
        .def("setName", &Imf::Header::setName)
        .def("type", [](Imf::Header& h) {return h.type();})
        .def("hasType", [](Imf::Header& h) {return h.hasType();})
        .def("setType", &Imf::Header::setType)

        .def("pixelType", [](Imf::Header& h) {
            auto channels = h.channels();
            Imf::PixelType pixel_type = channels.begin().channel().type;
            return pixel_type;
        })
        .def("getChannels", [](Imf::Header& h) { return getChannels(h); })
        .def("setChannels", [](Imf::Header& h, Imf::PixelType& pixel_type, py::list& channel_names) {
           for (auto it = channel_names.begin(); it != channel_names.end(); ++it) {
               h.channels().insert(it->cast<std::string>(), Imf::Channel(pixel_type));
           }
        })
        .def("getAttributes", [](Imf::Header& h) { return getAttributes(h);})
        //.def("getAttribute", [](Imf::Header& h) { return getAttribute(h);})
        .def("getCompression", [](Imf::Header& h) {return h.compression();})
        .def("setCompression", [](Imf::Header& h, Imf::Compression& c)
            { h.compression() = c; }
        )
        .def("getDWACompressionLevel", [](Imf::Header& h) {return h.dwaCompressionLevel();})
        .def("setDWACompressionLevel", [](Imf::Header& h, float& value)
            { h.dwaCompressionLevel() = value; }
        )
        .def("setAttribute", [](Imf::Header& h, const char* n, const char* v) {
            h.insert(n, Imf::StringAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const int v) {
            h.insert(n, Imf::IntAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const float v) {
            h.insert(n, Imf::FloatAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const double v) {
            h.insert(n, Imf::DoubleAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Box2i v) {
            h.insert(n, Imf::Box2iAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Box2f v) {
            h.insert(n, Imf::Box2fAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Vec3<int> v) {
            h.insert(n, Imf::V3iAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Vec3<float> v) {
            h.insert(n, Imf::V3fAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Vec3<double> v) {
            h.insert(n, Imf::V3dAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Vec2<int> v) {
            h.insert(n, Imf::V2iAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Vec2<float> v) {
            h.insert(n, Imf::V2fAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Vec2<double> v) {
            h.insert(n, Imf::V2dAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Matrix33<float> v) {
            h.insert(n, Imf::M33fAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Matrix33<double> v) {
            h.insert(n, Imf::M33dAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Matrix44<float> v) {
            h.insert(n, Imf::M44fAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imath::Matrix44<double> v) {
            h.insert(n, Imf::M44dAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imf::TimeCode v) {
            h.insert(n, Imf::TimeCodeAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imf::Rational v) {
            h.insert(n, Imf::RationalAttribute(v));
        })
        .def("setAttribute", [](Imf::Header& h, const char* n, const Imf::Compression v) {
            h.insert(n, Imf::CompressionAttribute(v));
        })
        ;


    py::class_<Imf::Name>(m, "Name");
    py::class_<Imf::Rational>(m, "Rational")
        .def(py::init())
        .def(py::init<int, int>(), py::arg("n"), py::arg("d"))
        .def(py::init<double>(), py::arg("value"))
        .def_readwrite("n", &Imf::Rational::n)
        .def_readwrite("d", &Imf::Rational::d)
        .def("__float__", [](const Imf::Rational& r) {
            return static_cast<double>(r);
        })
        .def("__repr__", [](const Imf::Rational& r) {
            std::stringstream ss;
            ss << "Rational(" << r.n << "/" << r.d << ")";
            return ss.str();
        })
        ;

    py::class_<Imf::TimeCode>(m, "TimeCode")
        .def(py::init())
        .def(py::init<int, int, int, int>(), py::arg("hours"), py::arg("minutes"), py::arg("seconds"), py::arg("frame"))
        .def(py::init<int>(), py::arg("timeAndFlags"))
        .def("hours", &Imf::TimeCode::hours)
        .def("minutes", &Imf::TimeCode::minutes)
        .def("seconds", &Imf::TimeCode::seconds)
        .def("frame", &Imf::TimeCode::frame)
        .def("timeAndFlags", &Imf::TimeCode::timeAndFlags)
        .def("setHours", &Imf::TimeCode::setHours)
        .def("setMinutes", &Imf::TimeCode::setMinutes)
        .def("setSeconds", &Imf::TimeCode::setSeconds)
        .def("setFrame", &Imf::TimeCode::setFrame)
        .def("setTimeAndFlags", &Imf::TimeCode::setTimeAndFlags)
        .def("__repr__", [](const Imf::TimeCode& r) {
            std::stringstream ss;
            ss << std::format("TimeCode({}:{}:{}:{})", r.hours(), r.minutes(), r.seconds(), r.frame());
            return ss.str();
        })
        ;

    py::class_<Imf::MultiPartInputFile>(m, "MultiPartInputFile")
        .def(py::init<const char*>(), py::arg("path"))
        .def("parts", &Imf::MultiPartInputFile::parts)
        .def("header", &Imf::MultiPartInputFile::header)
        .def("headers", [](Imf::MultiPartInputFile& f) { f;} )
        ;
        
    py::class_<Imf::MultiPartOutputFile>(m, "MultiPartOutputFile")
        // intialize using a lambda to convert an input list to an array pointer.
        .def(py::init([](const char* path, const py::list& headers) {
            std::vector<Imf::Header> hdrs = headers.cast<std::vector<Imf::Header>>();
            return new Imf::MultiPartOutputFile(
                path,
                hdrs.data(),
                hdrs.size()
            );
        }), py::arg("path"), py::arg("headers"))
        ;

    py::class_<Imf::TiledInputPart>(m, "TiledInputPart")
        .def(py::init<Imf::MultiPartInputFile&, int>(), py::arg("file"), py::arg("part") = py::int_(0))
        .def("header", &Imf::TiledInputPart::header)
        .def("numLevels", &Imf::TiledInputPart::numLevels)
        .def("levelWidth", &Imf::TiledInputPart::levelWidth)
        .def("levelHeight", &Imf::TiledInputPart::levelHeight)
        .def("readPixels", [](Imf::TiledInputPart& part, const int level_index = 0) {
            return readPartTiled(part, level_index);
        }) // py::arg("level_index") = py::int_(0)
        ;

    py::class_<Imf::TiledOutputPart>(m, "TiledOutputPart")
        .def(py::init<Imf::MultiPartOutputFile&, int>(), py::arg("file"), py::arg("part") = py::int_(0))
        .def("header", &Imf::TiledOutputPart::header)
        .def("copyPixels", [](Imf::TiledOutputPart& out_part, Imf::TiledInputPart& in_part) {
            out_part.copyPixels(in_part);
        })
        .def("writePixels", [](Imf::TiledOutputPart& part, py::array_t<half>& array) {
            writePartTiled<half>(part, array);
        })
        .def("writePixels", [](Imf::TiledOutputPart& part, py::array_t<float>& array) {
            writePartTiled<float>(part, array);
        })
        .def("writePixels", [](Imf::TiledOutputPart& part, py::array_t<uint32_t>& array) {
            writePartTiled<uint32_t>(part, array);
        })
        .def("recompress", [](Imf::TiledOutputPart& out_part, Imf::TiledInputPart& in_part) {
            recompressTiled(in_part, out_part);
        },
        "Recompresses the image data in the input part and writes the result to the output part."
        )
        ;

    py::class_<Imf::InputPart>(m, "InputPart")
        .def(py::init<Imf::MultiPartInputFile&, int>(), py::arg("file"), py::arg("part") = py::int_(0))
        .def("header", &Imf::InputPart::header)
        .def("readPixels", [](Imf::InputPart& part) {
            return readPart(part);
        })
        ;

    py::class_<Imf::OutputPart>(m, "OutputPart")
        .def(py::init<Imf::MultiPartOutputFile&, int>(), py::arg("file"), py::arg("part") = py::int_(0))
        .def("header", &Imf::OutputPart::header)
        .def("copyPixels", [](Imf::OutputPart& out_part, Imf::InputPart& in_part) {
            out_part.copyPixels(in_part);
        })
        .def("writePixels", [](Imf::OutputPart& part, py::array_t<half>& array) {
            writePart<half>(part, array);
        })
        .def("writePixels", [](Imf::OutputPart& part, py::array_t<float>& array) {
            writePart<float>(part, array);
        })
        .def("writePixels", [](Imf::OutputPart& part, py::array_t<uint32_t>& array) {
            writePart<uint32_t>(part, array);
        })
        .def("recompress", [](Imf::OutputPart& out_part, Imf::InputPart& in_part) {
            recompress(in_part, out_part);
        },
        "Recompresses the image data in the input part and writes the result to the output part."
        )
        ;
};
