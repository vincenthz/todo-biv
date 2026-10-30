#!/bin/sh
# Builds "Todo Biv.app" with cargo-bundle and signs it ad hoc, so macOS can
# ask for (and remember) microphone access.
#
#   scripts/bundle.sh            # target/release/bundle/osx/Todo Biv.app
#   scripts/bundle.sh --dmg      # and target/release/bundle/dmg/Todo Biv.dmg
#
# cargo-bundle resolves the icon and Info.plist.ext paths of
# [package.metadata.bundle] against the current directory, so it must run from
# crates/app: from the workspace root (`-p todo-biv`) it silently leaves them out.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
dmg=false
for arg in "$@"; do
    case "$arg" in
        --dmg) dmg=true ;;
        *) echo "usage: $0 [--dmg]" >&2; exit 2 ;;
    esac
done

if ! command -v cargo-bundle >/dev/null 2>&1; then
    echo "cargo-bundle is not installed: cargo install cargo-bundle" >&2
    exit 1
fi

bundles="${CARGO_TARGET_DIR:-$root/target}/release/bundle"
app="$bundles/osx/Todo Biv.app"

cd "$root/crates/app"
cargo bundle --release --format osx

# ad hoc: binds Info.plist (NSMicrophoneUsageDescription) to the code; without
# the hardened runtime no audio-input entitlement is needed
codesign --force --deep --sign - "$app"
codesign --verify --strict "$app"
echo "signed: $app"

if $dmg; then
    # cargo-bundle's dmg format would pack a fresh, unsigned copy of the app
    staging="$bundles/dmg/staging"
    image="$bundles/dmg/Todo Biv.dmg"
    rm -rf "$staging"
    mkdir -p "$staging"
    ditto "$app" "$staging/Todo Biv.app"
    ln -s /Applications "$staging/Applications"
    hdiutil create -quiet -ov -volname "Todo Biv" -srcfolder "$staging" -format UDZO "$image"
    rm -rf "$staging"
    echo "disk image: $image"
fi
