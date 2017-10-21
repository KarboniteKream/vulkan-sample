.PHONY: build
default: build

build:
	g++ -m64 -std=c++11 src/main.cpp -I vendor/vulkan/Include -I vendor/glfw/include -L vendor/vulkan/Lib -L vendor/glfw/lib -l glfw3 -l vulkan-1 -o build/main.exe
