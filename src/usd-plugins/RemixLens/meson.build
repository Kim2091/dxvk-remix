# RemixLens.dll registers the TfType of RemixLensAPI, so HasAPI() works.
# The properties come from the codeless schema in resources/, which is copied
# next to the DLL so scripts-common/install_usd_plugins.bat installs both.
remix_lens_plugin = shared_module(
  'RemixLens',
  'remixLensAPI.cpp',
  include_directories: include_directories('../' + usd_schema_plugins_nvusd_dir + '/include'),
  dependencies: usd_schema_plugins_linklibs,
  cpp_args: ['-DNOMINMAX', usd_schema_plugins_isdebug ? '-DTBB_USE_DEBUG=1' : '-DTBB_USE_DEBUG=0', '/Zc:inline-', '/wd4251', '/wd4275', '/wd4305', '/wd4244', '/wd4273'],
  override_options: ['cpp_std=c++17'],
  install: false,
)

configure_file(
  input: 'resources/plugInfo.json',
  output: 'plugInfo.json',
  copy: true,
)
configure_file(
  input: 'resources/generatedSchema.usda',
  output: 'generatedSchema.usda',
  copy: true,
)

# The resources folder mirrors the installed layout, so USD can load the build-tree plugin.
subdir('resources')
