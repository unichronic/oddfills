SHADERS_DIR = /home/unichronic/oddfills/anse/shaders
SPV_DIR     = /home/unichronic/oddfills/anse/build/shaders
GLSLANG     = glslangValidator

shaders:
	$(GLSLANG) -V $(SHADERS_DIR)/neural.vert -o $(SPV_DIR)/neural.vert.spv
	$(GLSLANG) -V $(SHADERS_DIR)/neural.frag -o $(SPV_DIR)/neural.frag.spv
	$(GLSLANG) -V $(SHADERS_DIR)/geometry.vert -o $(SPV_DIR)/geometry.vert.spv
	$(GLSLANG) -V $(SHADERS_DIR)/geometry.frag -o $(SPV_DIR)/geometry.frag.spv

all: shaders
	g++ -O2 vulk.cpp imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp \
	    imgui/imgui_widgets.cpp imgui/backends/imgui_impl_glfw.cpp \
	    imgui/backends/imgui_impl_vulkan.cpp \
	    -o oddfills -Iinclude -Iimgui -Iimgui/backends \
	    -lvulkan -lglfw -lpthread -ldl -lm

clean:
	rm -f oddfills vulk.cpp
