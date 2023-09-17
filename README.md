## How To Build:

You will need:

- [CMake](https://cmake.org/download/)
- [Python](https://www.python.org/) (added to your PATH)
- [Vulkan SDK](https://vulkan.lunarg.com/)

For desktop:
```
git submodule update --init --recursive
mkdir build
cd build
cmake ..
```
For Web:


Download [emscripten](https://emscripten.org/) and follow installation guide.


On Windows you may need to download [ninja](https://ninja-build.org/) and include the folder in your PATH environment variable, then:


```
git submodule update --init --recursive
mkdir build-web
cd build-web
emcmake cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

After building for web, you can host a local server with python:

```
python -m http.server
```

And access the webpage using a browser with WebGPU support using the link: ``localhost:8000/rooms.html``
