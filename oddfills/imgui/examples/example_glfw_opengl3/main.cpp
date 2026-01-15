#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION #if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif #include <GLFW/glfw3.h>
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif
#ifdef __EMSCRIPTEN__ #include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif
 static void glfw_error_callback(int error, const char* description)
 {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
 } int main(int, char**)
    {
  glfwSetErrorCallback(glfw_error_callback);
if (!glfwInit())
  return 1; #if defined(IMGUI_IMPL_OPENGL_ES2)
        const char* glsl_version = "#version 100"; glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0); glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__) const char* glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
 glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    const char* glsl_version = "#version 130";
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0); #endif
        GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == nullptr)
  return 1; glfwMakeContextCurrent(window);
 glfwSwapInterval(1); IMGUI_CHECKVERSION();
  ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
#ifdef __EMSCRIPTEN__
        ImGui_ImplGlfw_InstallEmscriptenCanvasResizeCallback("#canvas");
#endif
  ImGui_ImplOpenGL3_Init(glsl_version);
  bool show_demo_window = true; bool show_another_window = false;
ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
#ifdef __EMSCRIPTEN__
  io.IniFilename = nullptr; EMSCRIPTEN_MAINLOOP_BEGIN
#else
while (!glfwWindowShouldClose(window)) #endif {
    glfwPollEvents(); ImGui_ImplOpenGL3_NewFrame();
 ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);
    {
  static float f = 0.0f;
 static int counter = 0;
 ImGui::Begin("Hello, world!"); ImGui::Text("This is some useful text.");
        ImGui::Checkbox("Demo Window", &show_demo_window); ImGui::Checkbox("Another Window", &show_another_window);
 ImGui::SliderFloat("float", &f, 0.0f, 1.0f); ImGui::ColorEdit3("clear color", (float*)&clear_color);
 if (ImGui::Button("Button"))
 counter++;
ImGui::SameLine(); ImGui::Text("counter = %d", counter);
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
        }
if (show_another_window)
 { ImGui::Begin("Another Window", &show_another_window); ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
 show_another_window = false;
  ImGui::End(); } ImGui::Render();
  int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
 glViewport(0, 0, display_w, display_h);
  glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    } #ifdef __EMSCRIPTEN__
  EMSCRIPTEN_MAINLOOP_END;
#endif
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext();
    glfwDestroyWindow(window); glfwTerminate();
return 0; }