#pragma once

#include <variant>

#include "../../d3d9/d3d9_device.h"
#include "../../d3d11/d3d11_context_imm.h"

namespace dxvk {
  // Registration selects the frontend; all API scene commands use its stream.
  class RemixApiDevice {
  public:
    explicit RemixApiDevice(D3D9DeviceEx* device)
      : m_d3d9(device), m_device(device->GetDXVKDevice()) { }

    RemixApiDevice(D3D11ImmediateContext* context, const Rc<DxvkDevice>& device)
      : m_d3d11(context), m_device(device) { }

    const Rc<DxvkDevice>& GetDXVKDevice() const { return m_device; }
    D3D9DeviceEx* d3d9() const { return m_d3d9; }
    D3D11ImmediateContext* d3d11() const { return m_d3d11; }

    std::variant<D3D9DeviceLock, D3D10DeviceLock> LockDevice() {
      if (m_d3d11) {
        return m_d3d11->LockContext();
      }
      return m_d3d9->LockDevice();
    }

    template<bool Synchronize = true, typename Cmd>
    void EmitCs(Cmd&& command) {
      if (m_d3d11) {
        m_d3d11->EmitRemixCs(std::forward<Cmd>(command));
      } else {
        m_d3d9->EmitCs<Synchronize>(std::forward<Cmd>(command));
      }
    }

  private:
    D3D9DeviceEx* m_d3d9 = nullptr;
    D3D11ImmediateContext* m_d3d11 = nullptr;
    Rc<DxvkDevice> m_device;
  };
}
