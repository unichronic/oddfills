#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#include <stdio.h>
#include <SDL3/SDL.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL3/SDL_opengles2.h>
#else
#include <SDL3/SDL_opengl.h>
#endif
    int main(int, char**)
 {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMEPAD) != 0)
  {
printf("Error: SDL_Init(): %s\n", SDL_GetError());
  return -1;
}
 SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
 Uint32 window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN;
        SDL_Window* window = SDL_CreateWindow("Dear ImGui SDL3+SDL_Renderer example", 1280, 720, window_flags); if (window == nullptr)
 { printf("Error: SDL_CreateWindow(): %s\n", SDL_GetError()); return -1;
        } SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr, SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);
 if (renderer == nullptr)
        { SDL_Log("Error: SDL_CreateRenderer(): %s\n", SDL_GetError());
  return -1; }
 SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED); SDL_ShowWindow(window); IMGUI_CHECKVERSION();
        ImGui::CreateContext(); ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
 io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    ImGui::StyleColorsDark();
  ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
  ImGui_ImplSDLRenderer3_Init(renderer);
        bool show_demo_window = true;
  bool show_another_window = false; ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    bool done = false;
#ifdef __EMSCRIPTEN__
    io.IniFilename = nullptr; EMSCRIPTEN_MAINLOOP_BEGIN
#else
    while (!done)
#endif
  {
  SDL_Event event; while (SDL_PollEvent(&event)) { ImGui_ImplSDL3_ProcessEvent(&event);
if (event.type == SDL_EVENT_QUIT)
 done = true; if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window)) done = true; }
        ImGui_ImplSDLRenderer3_NewFrame();
  ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
if (show_demo_window)
        ImGui::ShowDemoWindow(&show_demo_window);
        {
        static float f = 0.0f; static int counter = 0;
    ImGui::Begin("Hello, world!"); ImGui::Text("This is some useful text.");
        ImGui::Checkbox("Demo Window", &show_demo_window); ImGui::Checkbox("Another Window", &show_another_window);
        ImGui::SliderFloat("float", &f, 0.0f, 1.0f); ImGui::ColorEdit3("clear color", (float*)&clear_color);
    if (ImGui::Button("Button"))
        counter++; ImGui::SameLine();
  ImGui::Text("counter = %d", counter); ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
ImGui::End();
  } if (show_another_window)
  { ImGui::Begin("Another Window", &show_another_window);
ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
        show_another_window = false;
    ImGui::End(); }
ImGui::Render();
  SDL_SetRenderDrawColor(renderer, (Uint8)(clear_color.x * 255), (Uint8)(clear_color.y * 255), (Uint8)(clear_color.z * 255), (Uint8)(clear_color.w * 255));
    SDL_RenderClear(renderer); ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData()); SDL_RenderPresent(renderer);
} ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
SDL_DestroyRenderer(renderer);
SDL_DestroyWindow(window); SDL_Quit();
        return 0; }