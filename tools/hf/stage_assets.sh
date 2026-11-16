#!/bin/bash
# Stage asset directories of the bucket mount on local disk (BUG-736), for job scripts:
#   source tools/hf/stage_assets.sh && stage_assets menagerie/franka_emika_panda mjcf/DamagedHelmet ...
# The mount has served files as empty when several processes of a loaded machine first read them at
# once (MuJoCo then fails with "Empty file 'panda.xml'" / "at least 4 vertices required"). Each
# directory is copied once, sequentially, and re-copied while any copied file is empty; every other
# entry of the asset root stays a link, and /work/prepared links the bucket's prepared/ (the
# prepared scenes are found next to the asset root). Exports MJPBR_ASSET_DIR=/work/assets.
stage_assets() {
  local root=${MJPBR_ASSET_DIR:-/data/assets} dst=/work/assets src d parent
  [ "$root" = "$dst" ] && root=/data/assets
  mkdir -p "$dst"
  for src in "$@"; do  # the parents of the staged directories become real directories of links
    parent=$(dirname "$src")
    [ "$parent" = . ] && continue
    [ -L "$dst/$parent" ] && rm -f "$dst/$parent"
    mkdir -p "$dst/$parent"
    for d in "$root/$parent"/*; do [ -e "$dst/$parent/$(basename "$d")" ] || ln -sfn "$d" "$dst/$parent/$(basename "$d")"; done
  done
  for d in "$root"/*; do [ -e "$dst/$(basename "$d")" ] || ln -sfn "$d" "$dst/$(basename "$d")"; done
  # The prepared scenes sit next to the asset root (<assets>/../prepared, where the scene registry looks).
  [ -e "$(dirname "$dst")/prepared" ] || ln -sfn "$(dirname "$root")/prepared" "$(dirname "$dst")/prepared"
  for src in "$@"; do
    rm -rf "$dst/$src"
    for try in 1 2 3 4; do
      cp -r "$root/$src" "$dst/$src.tmp" && rm -rf "$dst/$src" && mv "$dst/$src.tmp" "$dst/$src"
      [ -z "$(find "$dst/$src" -type f -size 0)" ] && break
      echo "STAGE_ASSETS empty files in $src (try $try): $(find "$dst/$src" -type f -size 0 | head -3 | tr '\n' ' ')"
      sleep 5
    done
  done
  export MJPBR_ASSET_DIR=$dst
}
