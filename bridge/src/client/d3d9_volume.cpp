/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "pch.h"
#include "d3d9_volume.h"

#include "util_bridge_assert.h"

/*
 * Direct3DVolume9_LSS Interface Implementation
 */

HRESULT Direct3DVolume9_LSS::QueryInterface(REFIID riid, LPVOID* ppvObj) {
  LogFunctionCall();
  if (ppvObj == nullptr) {
    return E_POINTER;
  }

  *ppvObj = nullptr;

  if (riid == __uuidof(IUnknown)
    || riid == __uuidof(IDirect3DVolume9)) {
    *ppvObj = bridge_cast<IDirect3DVolume9*>(this);
    AddRef();
    return S_OK;
  }

  return E_NOINTERFACE;
}

ULONG Direct3DVolume9_LSS::AddRef() {
  LogFunctionCall();
  // No push since we only care about the last Release call
  return D3DBase::AddRef();
}

ULONG Direct3DVolume9_LSS::Release() {
  LogFunctionCall();
  return D3DBase::Release();
}

void Direct3DVolume9_LSS::onDestroy() {
  // The standalone volumes use normal destroy command, however child volumes
  // are completely owned and managed by their parent container, and so only need
  // to be unlinked from x64 counterpart to prevent hash collisions at server side.
  const auto command = isStandalone() ? Commands::IDirect3DVolume9_Destroy :
    Commands::Bridge_UnlinkVolumeResource;

  ClientMessage { command, getId() };
}

HRESULT Direct3DVolume9_LSS::GetDevice(IDirect3DDevice9** ppDevice) {
  LogFunctionCall();
  m_pDevice->AddRef();
  (*ppDevice) = m_pDevice;
  return S_OK;
}

HRESULT Direct3DVolume9_LSS::SetPrivateData(REFGUID refguid, CONST void* pData, DWORD SizeOfData, DWORD Flags) {
  LogFunctionCall();
  return m_privateData.setData(refguid, pData, SizeOfData, Flags);
}

HRESULT Direct3DVolume9_LSS::GetPrivateData(REFGUID refguid, void* pData, DWORD* pSizeOfData) {
  LogFunctionCall();
  return m_privateData.getData(refguid, pData, pSizeOfData);
}

HRESULT Direct3DVolume9_LSS::FreePrivateData(REFGUID refguid) {
  LogFunctionCall();
  return m_privateData.freeData(refguid);
}

HRESULT Direct3DVolume9_LSS::GetContainer(REFIID riid, void** ppContainer) {
  LogFunctionCall();
  return getParent()->QueryInterface(riid, ppContainer);
}

HRESULT Direct3DVolume9_LSS::GetDesc(D3DVOLUME_DESC* pDesc) {
  LogFunctionCall();
  *pDesc = m_desc;
  return S_OK;
}

HRESULT Direct3DVolume9_LSS::LockBox(D3DLOCKED_BOX* pLockedVolume, CONST D3DBOX* pBox, DWORD Flags) {
  LogFunctionCall();
  // Store locked rect pointer locally so we can copy the data on unlock
  {
    BRIDGE_PARENT_DEVICE_LOCKGUARD();
    if (!lock(*pLockedVolume, pBox, Flags)) {
      std::stringstream ss;
      ss << "[Direct3DVolume9_LSS][LockBox] Failed!";
      Logger::err(ss.str());
      return E_FAIL;
    }
  }
  // Do nothing on lock on server side for now, all logic happens on unlock!
  return S_OK;
}

HRESULT Direct3DVolume9_LSS::UnlockBox() {
  LogFunctionCall();
  {
    BRIDGE_PARENT_DEVICE_LOCKGUARD();
    unlock();
  }
  return S_OK;
}

bool Direct3DVolume9_LSS::lock(D3DLOCKED_BOX& lockedVolume, const D3DBOX* const pBox, const DWORD flags) {
  const D3DBOX box = resolveLockInfoBox(pBox, m_desc);
  const auto [width, height, depth] = getBoxDimensions(box);
  // The pitches are in bytes, rows of blocks for compressed formats, with no padding between rows and slices.
  const size_t rowSize = bridge_util::calcRowSize((uint32_t) width, m_desc.Format);
  const size_t numRows = bridge_util::calcStride((uint32_t) height, m_desc.Format);
  void* const pData = new uint8_t[rowSize * numRows * depth];
  lockedVolume.RowPitch = (INT) rowSize;
  lockedVolume.SlicePitch = (INT) (rowSize * numRows);
  lockedVolume.pBits = pData;
  m_lockInfoQueue.push({ lockedVolume, box, flags });
  return true;
}

void Direct3DVolume9_LSS::unlock() {
  // Some game engines may attempt to Unlock a non-locked resource "just in case"
  if (m_lockInfoQueue.empty()) {
    return;
  }
  const auto lockInfo = m_lockInfoQueue.front();
  m_lockInfoQueue.pop();

  // Prep command send
  const auto& lockedVolume = lockInfo.lockedVolume;
  const auto& box = lockInfo.box;
  const uint32_t depth = (uint32_t) std::get<2>(getBoxDimensions(box));
  const uint32_t rowSize = (uint32_t) lockedVolume.RowPitch;
  const uint32_t numRows = (uint32_t) (lockedVolume.SlicePitch / lockedVolume.RowPitch);

  // Now send the box dimensions and surface handle
  if ((lockInfo.flags & D3DLOCK_READONLY) == 0) {
    ClientMessage c(Commands::IDirect3DVolume9_UnlockBox, getId());

    c.send_data(sizeof(D3DBOX), &box);
    c.send_data(lockInfo.flags);

    // The rows of all slices follow in order, in blobs of whole rows of at most one item each.
    // The server writes them into the volume under one lock.
    c.send_many(rowSize, numRows, depth);
    const uint64_t totalRows = uint64_t(numRows) * depth;
    const uint64_t rowsPerBlob = std::max<uint64_t>(1, DeviceBridge::maxItemBytes() / rowSize);
    for (uint64_t row = 0; row < totalRows && gbBridgeRunning;) {
      const uint64_t blobRows = std::min(rowsPerBlob, totalRows - row);
      c.send_data((uint32_t) (blobRows * rowSize), static_cast<const uint8_t*>(lockedVolume.pBits) + row * rowSize);
      row += blobRows;
    }
  }
  delete[] static_cast<uint8_t*>(lockedVolume.pBits);
}

D3DBOX Direct3DVolume9_LSS::resolveLockInfoBox(const D3DBOX* const pBox, const D3DVOLUME_DESC& desc) {
  return (pBox != nullptr) ? *pBox : D3DBOX { 0, 0, desc.Width, desc.Height, 0, desc.Depth };
}

std::tuple<size_t, size_t, size_t> Direct3DVolume9_LSS::getBoxDimensions(const D3DBOX& box) {
  return { box.Right  - box.Left,
           box.Bottom - box.Top,
           box.Back   - box.Front };
}
