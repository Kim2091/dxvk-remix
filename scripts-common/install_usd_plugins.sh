#!/usr/bin/env bash

set -e
shopt -s nullglob

src_base="${MESON_BUILD_ROOT}/src/usd-plugins"
dst_base="$1/usd/plugins"

mkdir -p "${dst_base}"

for plugin_dir in "${src_base}"/*/; do
  plugin_dir=${plugin_dir%/}
  if [[ ! -f "${plugin_dir}/plugInfo.json" ]]; then
    continue
  fi
  dst="${dst_base}/$(basename "${plugin_dir}")"
  mkdir -p "${dst}/resources"

  cp -f "${plugin_dir}/plugInfo.json" "${dst}/resources/"

  for library in "${plugin_dir}"/*.dll; do
    cp -f "${library}" "${dst}/"
  done

  # generatedSchema.usda lives at the plugin root (ResourcePath ".").
  if [[ -f "${plugin_dir}/generatedSchema.usda" ]]; then
    cp -f "${plugin_dir}/generatedSchema.usda" "${dst}/"
  fi
done
