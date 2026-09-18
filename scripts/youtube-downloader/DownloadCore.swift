import Foundation

enum OutputFormat: String, CaseIterable {
    case mp4, mkv, m4a, mp3
}

struct DownloadSettings {
    var height: Int? = 1080
    var format: OutputFormat = .mp4
    var subtitles: Bool = false
    var thumbnail: Bool = false
    var browser: String? = nil
}

private enum DownloadCoreError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        }
    }
}

struct DownloadTools {
    let ytDlp: String
    let ffmpeg: String
    let javascriptRuntime: String
    let javascriptPath: String

    static func discover(preferredDirectory: URL? = nil) throws -> DownloadTools {
        guard let downloader = findExecutable("yt-dlp", preferredDirectory: preferredDirectory) else {
            throw DownloadCoreError.invalid("Не найден yt-dlp. Выполните в Терминале: brew install yt-dlp ffmpeg node")
        }
        guard let ffmpeg = findExecutable("ffmpeg") else {
            throw DownloadCoreError.invalid("Не найден ffmpeg. Выполните в Терминале: brew install ffmpeg")
        }
        let ffprobe = URL(fileURLWithPath: ffmpeg).deletingLastPathComponent().appendingPathComponent("ffprobe").path
        guard FileManager.default.isExecutableFile(atPath: ffprobe) else {
            throw DownloadCoreError.invalid("Рядом с ffmpeg не найден ffprobe. Переустановите пакет: brew reinstall ffmpeg")
        }
        for runtime in ["node", "deno"] {
            if let path = findExecutable(runtime) {
                return DownloadTools(ytDlp: downloader, ffmpeg: ffmpeg, javascriptRuntime: runtime, javascriptPath: path)
            }
        }
        throw DownloadCoreError.invalid("Для YouTube нужен Node.js или Deno. Выполните в Терминале: brew install node")
    }

    private static func findExecutable(_ name: String, preferredDirectory: URL? = nil) -> String? {
        let preferred = preferredDirectory.flatMap { $0.isFileURL ? $0.path : nil }
        let directories = preferred.map { [$0] } ?? []
        for directory in directories + executableDirectories() {
            let path = URL(fileURLWithPath: directory, isDirectory: true).appendingPathComponent(name).path
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
               !isDirectory.boolValue, FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }
        return nil
    }
}

struct DownloadCommand {
    let executable: String
    let arguments: [String]
    let environment: [String: String]
}

private func executableDirectories() -> [String] {
    // Finder does not inherit the interactive shell's Homebrew PATH.
    let inherited = ProcessInfo.processInfo.environment["PATH", default: ""].split(separator: ":").map(String.init)
    let candidates = inherited + [
        "/opt/homebrew/bin", "/usr/local/bin",
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin").path,
        "/usr/bin", "/bin", "/usr/sbin", "/sbin",
    ]
    var seen: Set<String> = []
    return candidates.filter { $0.hasPrefix("/") && seen.insert($0).inserted }
}

/// Accept individual YouTube video links only; queue expansion never happens in yt-dlp.
func validatedVideoURLs(_ text: String) throws -> [String] {
    guard text.utf8.count <= 65_536 else {
        throw DownloadCoreError.invalid("Слишком большой список ссылок. Добавьте до 50 видео за один раз.")
    }
    let tokens = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    guard !tokens.isEmpty else {
        throw DownloadCoreError.invalid("Вставьте хотя бы одну ссылку на видео YouTube.")
    }
    guard tokens.count <= 50 else {
        throw DownloadCoreError.invalid("Добавьте не больше 50 ссылок за один раз.")
    }

    let youtubeHosts: Set<String> = ["youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"]
    let shortHosts: Set<String> = ["youtu.be", "www.youtu.be"]
    var seen: Set<String> = []
    var result: [String] = []
    for (index, token) in tokens.enumerated() {
        let candidate = token.contains("://") ? token : "https://" + token
        guard let components = URLComponents(string: candidate),
              let scheme = components.scheme?.lowercased(), ["http", "https"].contains(scheme),
              let host = components.host?.lowercased(),
              components.user == nil, components.password == nil, components.port == nil else {
            throw invalidLink(index)
        }
        let parts = components.path.split(separator: "/", omittingEmptySubsequences: false)
        var videoID: String?
        if shortHosts.contains(host), parts.count == 2, parts[0].isEmpty {
            videoID = String(parts[1])
        } else if youtubeHosts.contains(host) {
            if components.path == "/watch" {
                let ids = (components.queryItems ?? []).filter { $0.name == "v" }
                if ids.count == 1 { videoID = ids[0].value }
            } else if parts.count == 3, parts[0].isEmpty,
                      ["shorts", "live", "embed"].contains(String(parts[1])) {
                videoID = String(parts[2])
            }
        }
        guard let id = videoID, id.utf8.count == 11,
              id.range(of: "^[A-Za-z0-9_-]{11}$", options: .regularExpression) != nil else {
            throw invalidLink(index)
        }
        let canonical = "https://www.youtube.com/watch?v=\(id)"
        if seen.insert(canonical).inserted { result.append(canonical) }
    }
    return result
}

private func invalidLink(_ index: Int) -> DownloadCoreError {
    .invalid("Ссылка \(index + 1): нужна прямая ссылка на видео YouTube (watch, youtu.be, Shorts или live). Плейлисты и каналы не поддерживаются.")
}

/// Build argv directly. Neither pasted text nor the output directory is shell code.
func makeDownloadCommand(url: String, settings: DownloadSettings, directory: URL, tools: DownloadTools) throws -> DownloadCommand {
    let urls = try validatedVideoURLs(url)
    guard urls.count == 1 else { throw DownloadCoreError.invalid("Команда должна содержать ровно одно видео.") }
    guard directory.isFileURL else { throw DownloadCoreError.invalid("Для сохранения нужна локальная папка.") }
    if let height = settings.height, ![144, 240, 360, 480, 720, 1080, 1440, 2160, 4320].contains(height) {
        throw DownloadCoreError.invalid("Выберите доступное разрешение видео.")
    }
    if let browser = settings.browser, !["arc", "chrome", "safari", "firefox"].contains(browser) {
        throw DownloadCoreError.invalid("Поддерживаются cookies из Arc, Chrome, Safari или Firefox.")
    }
    guard ["node", "deno"].contains(tools.javascriptRuntime) else {
        throw DownloadCoreError.invalid("Выберите Node.js или Deno для обработки ссылок YouTube.")
    }

    let isAudio = settings.format == .m4a || settings.format == .mp3
    let quality = isAudio ? "audio" : settings.height.map { "\($0)p" } ?? "best"
    let filename = "%(title).120B [%(id)s] [\(quality)-\(settings.format.rawValue)].%(ext)s"
    var args = [
        "--ignore-config", "--no-plugin-dirs", "--no-remote-components",
        "--no-js-runtimes", "--js-runtimes", "\(tools.javascriptRuntime):\(tools.javascriptPath)",
        "--ffmpeg-location", tools.ffmpeg,
        "--no-playlist", "--no-overwrites", "--no-post-overwrites", "--continue",
        "--no-cache-dir", "--newline", "--no-color", "--no-simulate", "--no-quiet", "--progress",
        "--progress-delta", "0.5",
        "--progress-template", "download:VS_PROGRESS|%(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s",
        "--print", "after_move:VS_SAVED|%(filepath)s",
        "--socket-timeout", "30", "--retries", "3", "--fragment-retries", "3",
        "--extractor-retries", "2", "--file-access-retries", "2",
        "--retry-sleep", "http:2", "--retry-sleep", "fragment:2", "--retry-sleep", "extractor:2",
        "--abort-on-unavailable-fragments", "--no-wait-for-video",
        "--match-filters", "!is_live & live_status!=is_upcoming",
        "--windows-filenames", "-P", directory.path, "-o", filename,
    ]
    if isAudio {
        args += [
            "--format", settings.format == .m4a ? "ba[ext=m4a]/ba/b" : "ba/b",
            "--extract-audio", "--audio-format", settings.format.rawValue, "--audio-quality", "0",
        ]
    } else {
        let cap = settings.height.map { "[height<=\($0)]" } ?? ""
        // Prefer AAC for MP4, without restricting video to codecs that stop at 1080p.
        let selector = settings.format == .mp4
            ? "bv*\(cap)+ba[ext=m4a]/bv*\(cap)+ba/b\(cap)"
            : "bv*\(cap)+ba/b\(cap)"
        args += ["--format", selector, "--format-sort", "res,fps,vcodec:h264,acodec:aac",
                 "--merge-output-format", settings.format.rawValue, "--remux-video", settings.format.rawValue]
    }
    if settings.subtitles {
        args += ["--write-subs", "--write-auto-subs", "--sub-langs", "ru.*,en.*", "--sub-format", "vtt/best"]
    }
    if settings.thumbnail { args.append("--write-thumbnail") }
    if let browser = settings.browser {
        // yt-dlp has no native "arc" reader. Arc uses Chromium's cookie format
        // but keeps profiles separately from both Chrome and Chromium.
        let profile = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Arc/User Data").path
        let source = browser == "arc" ? "chromium:\(profile)" : browser
        args += ["--cookies-from-browser", source]
    }
    args += ["--", urls[0]]

    var environment = ProcessInfo.processInfo.environment
    let toolDirectories = [tools.ytDlp, tools.ffmpeg, tools.javascriptPath].map {
        URL(fileURLWithPath: $0).deletingLastPathComponent().path
    }
    var seen: Set<String> = []
    environment["PATH"] = (toolDirectories + executableDirectories()).filter { seen.insert($0).inserted }.joined(separator: ":")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["LC_ALL"] = "en_US.UTF-8"
    return DownloadCommand(executable: tools.ytDlp, arguments: args, environment: environment)
}
