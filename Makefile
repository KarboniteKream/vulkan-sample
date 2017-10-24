.PHONY: build shaders

all: build shaders

build:
	g++ -std=c++11 -m64 -g src/main.cpp -I vendor/vulkan/Include -I vendor/glfw/include -I vendor/glm -I vendor/stb -I vendor/tinyobjloader -L vendor/vulkan/Lib -L vendor/glfw/lib -l glfw3 -l vulkan-1 -o build/main.exe

release:
	g++ -std=c++11 -m64 -O3 -s src/main.cpp -I vendor/vulkan/Include -I vendor/glfw/include -I vendor/glm -I vendor/stb -I vendor/tinyobjloader -L vendor/vulkan/Lib -L vendor/glfw/lib -l glfw3 -l vulkan-1 -o build/main.exe

shaders:
	vendor/vulkan/Bin/glslangValidator -V shaders/shader.vert -o build/shaders/vert.spv
	vendor/vulkan/Bin/glslangValidator -V shaders/shader.frag -o build/shaders/frag.spv
