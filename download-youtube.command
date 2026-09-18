#!/bin/zsh

# Finder launches .command files in Terminal. Keep every generated file local.
set -euo pipefail
launcher_path="${0:A}"
project_root="${launcher_path:h}"
export PATH="$project_root/.youtube-downloader/runtime/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
source_dir="$project_root/scripts/youtube-downloader"
state_dir="$project_root/.youtube-downloader"
app_bundle="$state_dir/YouTube Downloader.app"
check_only=0
build_dir=""
backup_bundle=""

finish() {
    local result="$1"
    if [[ -n "$backup_bundle" && -d "$backup_bundle" && ! -e "$app_bundle" ]]; then
        /bin/mv -- "$backup_bundle" "$app_bundle"
    fi
    if [[ -n "$build_dir" && -d "$build_dir" ]]; then
        /bin/rm -rf -- "$build_dir"
    fi
    if (( result != 0 )); then
        print -u2 -- "Не удалось запустить загрузчик (код $result). Подробности — выше."
        if (( ! check_only )) && [[ -t 0 ]]; then
            read -r "?Нажмите Enter, чтобы закрыть окно…" || true
        fi
    fi
}
trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

if (( $# > 0 )); then
    if (( $# == 1 )) && [[ "$1" == "--check" ]]; then
        check_only=1
    else
        print -u2 -- "Использование: $0 [--check]"
        exit 2
    fi
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
    print -u2 -- "Это приложение работает только на macOS."
    exit 1
fi

missing_tools=()
for tool in yt-dlp ffmpeg ffprobe node; do
    command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool")
done
if (( ${#missing_tools[@]} > 0 )); then
    print -u2 -- "Не найдены программы: ${missing_tools[*]}"
    print -u2 -- "Установите зависимости через Homebrew: brew install yt-dlp ffmpeg node"
    exit 1
fi
if ! compiler_version="$(/usr/bin/xcrun swiftc --version 2>&1)"; then
    print -u2 -- "$compiler_version"
    print -u2 -- "Для сборки нужен Swift. Установите инструменты Apple: xcode-select --install"
    exit 1
fi
for source_file in DownloadCore.swift CommandRunner.swift main.swift; do
    if [[ ! -f "$source_dir/$source_file" ]]; then
        print -u2 -- "Не найден исходный файл: $source_dir/$source_file"
        exit 1
    fi
done

/bin/mkdir -p "$state_dir/module-cache" "$project_root/Данные матчи"

# flock releases its lock even when Terminal is closed during compilation.
zmodload zsh/system
touch "$state_dir/build.lock"
if ! zsystem flock -t 10 -f build_lock "$state_dir/build.lock"; then
    print -u2 -- "Другая копия ещё собирает приложение. Попробуйте открыть его снова через минуту."
    exit 1
fi

build_fingerprint() {
    {
        /usr/bin/shasum -a 256 "$source_dir/DownloadCore.swift" "$source_dir/CommandRunner.swift" "$source_dir/main.swift" "$launcher_path"
        print -r -- "$compiler_version"
        uname -m
    } | /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'
}

fingerprint="$(build_fingerprint)"
cached_fingerprint=""
if [[ -f "$app_bundle/Contents/Resources/build-fingerprint" ]]; then
    cached_fingerprint="$(<"$app_bundle/Contents/Resources/build-fingerprint")"
fi
if [[ "$fingerprint" != "$cached_fingerprint" || ! -x "$app_bundle/Contents/MacOS/YouTubeDownloader" ]]; then
    print -- "Собираю окно загрузчика. Первый запуск может занять несколько секунд…"
    build_dir="$(/usr/bin/mktemp -d "$state_dir/build.XXXXXXXX")"
    candidate="$build_dir/YouTube Downloader.app"
    /bin/mkdir -p "$candidate/Contents/MacOS" "$candidate/Contents/Resources"
    /usr/bin/xcrun swiftc -swift-version 5 -O \
        -module-cache-path "$state_dir/module-cache" \
        "$source_dir/DownloadCore.swift" "$source_dir/CommandRunner.swift" "$source_dir/main.swift" \
        -o "$candidate/Contents/MacOS/YouTubeDownloader"
    cat > "$candidate/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>YouTubeDownloader</string>
    <key>CFBundleIdentifier</key><string>local.videoscope.youtube-downloader</string>
    <key>CFBundleName</key><string>YouTube Downloader</string>
    <key>CFBundleDisplayName</key><string>Скачать с YouTube</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
    /usr/bin/plutil -lint "$candidate/Contents/Info.plist" >/dev/null
    if [[ "$fingerprint" != "$(build_fingerprint)" ]]; then
        print -u2 -- "Исходники изменились во время сборки. Откройте загрузчик повторно."
        exit 1
    fi
    print -r -- "$fingerprint" > "$candidate/Contents/Resources/build-fingerprint"
    /usr/bin/codesign --force --sign - "$candidate"
    /usr/bin/codesign --verify --strict "$candidate"
    if [[ -e "$app_bundle" ]]; then
        backup_bundle="$build_dir/previous.app"
        /bin/mv -- "$app_bundle" "$backup_bundle"
    fi
    if ! /bin/mv -- "$candidate" "$app_bundle"; then
        if [[ -n "$backup_bundle" ]]; then
            /bin/mv -- "$backup_bundle" "$app_bundle"
            backup_bundle=""
        fi
        exit 1
    fi
    backup_bundle=""
fi

if (( check_only )); then
    print -- "Готово: приложение собрано; yt-dlp, ffmpeg, ffprobe и Node.js найдены."
    print -- "Видео сохраняются в: $project_root/Данные матчи"
else
    /usr/bin/open -a "$app_bundle" --args "$project_root"
fi
