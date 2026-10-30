/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

// Dev-menu panels registered by frontends. The backend's dev menu cannot include frontend headers,
// so a frontend hands it a draw function instead.

namespace dxvk {
  namespace frontend_ui {

    enum class Group : unsigned {
      Geometry,  // Rendering > Geometry, after the D3D9 draw path controls
      Count
    };

    using DrawFn = void (*)();

    // Any thread. Registering the same function twice for a group draws it once.
    void registerPanel(Group group, DrawFn fn);
    // ImGui thread: draws the group's panels in registration order.
    void drawPanels(Group group);

  } // namespace frontend_ui
} // namespace dxvk
