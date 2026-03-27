# Arcane Wild Rune Engine

It reads any audio file and generates a glowing, porous 3D rune sphere in real time. It works by analyzing the audio frequencies, feeding them into a self-organizing neural network to shape the structure, and running a reaction-diffusion simulation to carve organic holes — inspired by the Wild Runes from Arcane.

## Installation

You need Vulkan, GLFW, and a C++ compiler. 
If you are on Linux (Ubuntu/Debian):

```bash
sudo apt install libvulkan-dev vulkan-tools libglfw3-dev build-essential glslang-tools
```

## How to use

1. Go into the folder:
```bash
cd oddfills
```

2. Build the project:
```bash
make
```

3. Run it:
```bash
./oddfills
```

Click the **Load Audio** button at the bottom and pick any `.wav` or `.mp3` file. The rune will start reacting immediately. Left click and drag to orbit the camera, scroll to zoom.
