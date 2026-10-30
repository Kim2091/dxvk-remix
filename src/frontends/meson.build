# Frontends. Compiled into the module of the API they intercept; the backend (src/dxvk)
# never includes anything from here, which the check below enforces at configure time.

run_command(python_interpreter, global_src_root_norm + '/scripts-common/check_frontend_deps.py', global_src_root_norm,
  check : true)

# Shared translator toolkit (frontends/translator/). Only compiled when a frontend that uses it is enabled.
frontend_translator_src = files([
  'translator/resource_stamp.h',
  'translator/material_translator.cpp',
  'translator/material_translator.h',
])

# D3D9 frontend. The facade always builds with D3D9 (it routes every draw); D3D9 v2 and the toolkit only with
# -Dfrontend_d3d9_v2=true.
frontend_d3d9_src = files([
  'd3d9/d3d9_draw_inputs.h',
  'd3d9/d3d9_frontend.cpp',
  'd3d9/d3d9_frontend.h',
])
frontend_d3d9_args = []
if get_option('frontend_d3d9_v2')
  frontend_d3d9_src += frontend_translator_src
  frontend_d3d9_args += [ '-DREMIX_FRONTEND_D3D9_V2=1' ]
endif
