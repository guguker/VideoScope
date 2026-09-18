import Foundation

@main
struct CoreTests {
    static func main() throws {
        var assertions = 0
        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            assertions += 1
            guard condition() else { fatalError(message) }
        }
        func rejected(_ text: String) {
            do {
                _ = try validatedVideoURLs(text)
                fatalError("Unexpectedly accepted: \(text)")
            } catch {
                assertions += 1
            }
        }
        let canonical = "https://www.youtube.com/watch?v=abcdefghijk"
        let variants = [
            "https://youtu.be/abcdefghijk?si=tracking",
            "https://www.youtube.com/watch?v=abcdefghijk&list=PL123&index=3&t=42",
            "https://m.youtube.com/shorts/abcdefghijk",
            "https://youtube.com/live/abcdefghijk?feature=share",
            "https://www.youtube.com/embed/abcdefghijk",
            "http://music.youtube.com/watch?v=abcdefghijk",
            "www.youtube.com/watch?v=abcdefghijk",
            "youtu.be/abcdefghijk",
        ]
        for variant in variants {
            let result = try validatedVideoURLs(variant)
            check(result == [canonical], "Canonicalization failed: \(variant)")
        }
        let deduplicated = try validatedVideoURLs(variants.joined(separator: "\n") + "\nhttps://youtu.be/ABCDEFGHIJK")
        check(deduplicated == [canonical, "https://www.youtube.com/watch?v=ABCDEFGHIJK"], "Queue order or deduplication failed")
        for invalid in [
            "", "   \n\t", "abcdefghijk", "--exec=/tmp/evil",
            "https://www.youtube.com/playlist?list=PL123",
            "https://www.youtube.com/@channel", "https://youtube.com/channel/abcdefghijk",
            "https://youtube.com.evil.example/watch?v=abcdefghijk",
            "https://evilyoutube.com/watch?v=abcdefghijk",
            "https://www.youtube.com@evil.example/watch?v=abcdefghijk",
            "https://someone@www.youtube.com/watch?v=abcdefghijk",
            "https://www.youtube.com:8443/watch?v=abcdefghijk",
            "file:///tmp/watch?v=abcdefghijk", "ftp://youtube.com/watch?v=abcdefghijk",
            "https://youtu.be/abcdefghijk/extra", "https://youtube.com/embed/abcdefghijk/extra",
            "https://youtube.com/watch?v=too-short", "https://youtube.com/watch?v=abcdefghijkl",
            "https://youtube.com/watch?v=abcdefghij!",
            "https://youtube.com/watch?v=abcdefghijk&v=ABCDEFGHIJK",
            "https://youtube.com/watch?v=abcdefghijk%0A--exec=evil",
            "https://youtube.com/watch?v=abcdefghijk%0A",
            "https://youtube.com/redirect?q=https://youtu.be/abcdefghijk",
            "https://youtu.be/abcdefghijk\n$(touch /tmp/never)",
            Array(repeating: canonical, count: 51).joined(separator: "\n"),
        ] { rejected(invalid) }
        let fifty = try validatedVideoURLs(Array(repeating: canonical, count: 50).joined(separator: "\n"))
        check(fifty == [canonical], "Valid batch limit rejected")

        let tools = DownloadTools(ytDlp: "/tools/yt-dlp", ffmpeg: "/tools/ffmpeg", javascriptRuntime: "node", javascriptPath: "/tools/node")
        let folder = URL(fileURLWithPath: "/tmp/Данные матчи $(literal)", isDirectory: true)
        func option(_ flag: String, in command: DownloadCommand) -> String? {
            guard let index = command.arguments.firstIndex(of: flag), index + 1 < command.arguments.count else { return nil }
            return command.arguments[index + 1]
        }
        let command = try makeDownloadCommand(url: variants[0], settings: DownloadSettings(), directory: folder, tools: tools)
        check(command.executable == tools.ytDlp, "Unexpected executable")
        check(option("-P", in: command) == folder.path, "Destination was shell-quoted or split")
        check(command.arguments.suffix(2) == ["--", canonical], "URL not canonicalized or option separator absent")
        for flag in ["--ignore-config", "--no-plugin-dirs", "--no-remote-components", "--no-js-runtimes", "--no-playlist", "--no-overwrites", "--no-post-overwrites", "--continue", "--newline", "--no-color", "--no-simulate", "--no-quiet", "--no-cache-dir", "--abort-on-unavailable-fragments", "--no-wait-for-video", "--progress"] {
            check(command.arguments.contains(flag), "Missing download boundary: \(flag)")
        }
        check(option("--js-runtimes", in: command) == "node:/tools/node", "Runtime not pinned")
        check(option("--ffmpeg-location", in: command) == tools.ffmpeg, "FFmpeg not explicit")
        check(option("--merge-output-format", in: command) == "mp4", "MP4 merge missing")
        check(option("--remux-video", in: command) == "mp4", "MP4 final container not enforced")
        check(option("--format", in: command)?.contains("height<=1080") == true, "Height cap missing")
        check(option("--format", in: command)?.contains("vcodec=h264") == false, "Codec restriction can discard higher resolution")
        check(option("--format-sort", in: command)?.hasPrefix("res") == true, "Resolution must precede codec preference")
        check(option("-o", in: command)?.contains("%(title).120B") == true, "Title byte limit missing")
        check(option("-o", in: command)?.contains("[1080p-mp4]") == true, "Quality/format profile missing")
        check(option("--match-filters", in: command)?.contains("is_upcoming") == true, "Upcoming-stream rejection missing")
        check(option("--match-filters", in: command)?.contains("is_live") == true, "Live-stream rejection missing")
        check(option("--progress-template", in: command) == "download:VS_PROGRESS|%(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s", "Progress protocol mismatch")
        check(option("--print", in: command) == "after_move:VS_SAVED|%(filepath)s", "Completion protocol mismatch")
        check(!command.arguments.contains("--cookies-from-browser"), "Cookies enabled by default")
        check(!command.arguments.contains("--download-archive"), "Archive interferes with quality changes")
        check(command.environment["PATH"]?.contains("/tools") == true, "Tool PATH absent")

        var names: Set<String> = []
        for format in OutputFormat.allCases {
            for height: Int? in [nil, 720, 1080, 2160] {
                var settings = DownloadSettings()
                settings.format = format
                settings.height = height
                let candidate = try makeDownloadCommand(url: canonical, settings: settings, directory: folder, tools: tools)
                let filename = option("-o", in: candidate)!
                if format == .mp4 || format == .mkv {
                    check(names.insert(filename).inserted, "Video quality profiles collide")
                    check(option("--remux-video", in: candidate) == format.rawValue, "Wrong video container")
                    if height == nil { check(!option("--format", in: candidate)!.contains("height"), "Best quality retains cap") }
                } else {
                    check(candidate.arguments.contains("--extract-audio"), "Audio extraction missing")
                    check(option("--audio-format", in: candidate) == format.rawValue, "Wrong audio format")
                    check(filename.contains("[audio-\(format.rawValue)]"), "Audio filename missing profile")
                    check(!option("--format", in: candidate)!.contains("height"), "Audio affected by video cap")
                }
            }
        }
        for browser in ["chrome", "safari", "firefox"] {
            var settings = DownloadSettings()
            settings.browser = browser
            settings.subtitles = true
            settings.thumbnail = true
            let selected = try makeDownloadCommand(url: canonical, settings: settings, directory: folder, tools: tools)
            check(option("--cookies-from-browser", in: selected) == browser, "Explicit browser not selected")
            check(selected.arguments.contains("--write-subs") && selected.arguments.contains("--write-auto-subs"), "Subtitle sources incomplete")
            check(option("--sub-langs", in: selected) == "ru.*,en.*", "Wrong subtitle languages")
            check(selected.arguments.contains("--write-thumbnail"), "Thumbnail option missing")
            check(!selected.arguments.contains("--embed-subs"), "Subtitles should remain sidecars")
        }
        var arc = DownloadSettings()
        arc.browser = "arc"
        let arcCommand = try makeDownloadCommand(url: canonical, settings: arc, directory: folder, tools: tools)
        let arcProfile = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Arc/User Data").path
        check(option("--cookies-from-browser", in: arcCommand) == "chromium:\(arcProfile)", "Arc must use its own profile with the Chromium cookie reader")
        check(arcCommand.arguments.filter { $0 == "--cookies-from-browser" }.count == 1, "Arc cookie reader was duplicated")
        check(!arcCommand.arguments.contains("--cookies"), "Arc must not export cookies to a file")
        var invalid = DownloadSettings()
        invalid.browser = "chrome:/tmp/profile"
        do {
            _ = try makeDownloadCommand(url: canonical, settings: invalid, directory: folder, tools: tools)
            fatalError("Unvalidated browser accepted")
        } catch { assertions += 1 }
        invalid = DownloadSettings()
        invalid.height = -1
        do {
            _ = try makeDownloadCommand(url: canonical, settings: invalid, directory: folder, tools: tools)
            fatalError("Invalid height accepted")
        } catch { assertions += 1 }
        do {
            _ = try makeDownloadCommand(url: canonical, settings: DownloadSettings(), directory: URL(string: "https://example.com/")!, tools: tools)
            fatalError("Remote destination accepted")
        } catch { assertions += 1 }

        // Discovery never launches these tools; keep the suite usable without Homebrew.
        if let baseline = try? DownloadTools.discover() {
            let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("downloader-discovery-\(UUID().uuidString)", isDirectory: true)
            let missing = try DownloadTools.discover(preferredDirectory: temporary)
            check(missing.ytDlp == baseline.ytDlp, "Missing preferred directory did not fall back")
            let remote = try DownloadTools.discover(preferredDirectory: URL(string: "https://example.com/runtime")!)
            check(remote.ytDlp == baseline.ytDlp, "Remote preferred directory should be ignored")
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: temporary) }
            let stub = temporary.appendingPathComponent("yt-dlp")
            try Data().write(to: stub)
            try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: stub.path)
            let notExecutable = try DownloadTools.discover(preferredDirectory: temporary)
            check(notExecutable.ytDlp == baseline.ytDlp, "Nonexecutable preferred file did not fall back")
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: stub.path)
            let preferred = try DownloadTools.discover(preferredDirectory: temporary)
            check(preferred.ytDlp == stub.path, "Executable in preferred directory was not selected")
            check(preferred.ffmpeg == baseline.ffmpeg, "Preferred directory changed FFmpeg discovery")
            check(preferred.javascriptRuntime == baseline.javascriptRuntime, "Preferred directory changed runtime selection")
            check(preferred.javascriptPath == baseline.javascriptPath, "Preferred directory changed runtime discovery")
        } else {
            print("Skipped installed-tool discovery checks: yt-dlp, FFmpeg/ffprobe, or Node/Deno unavailable")
        }
        print("Passed \(assertions) offline core assertions")
    }
}
